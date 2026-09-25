import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.assembler import assemble_full_dub, process_and_trim_chunk
from src.diarizer import add_new_character, cluster_speakers_for_clip, get_characters, update_character
from src.downloader import download_youtube_clip
from src.segmenter import get_or_create_segments, save_segments
from src.separator import separate_audio_stems

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Dubalike — Voice Dubbing & Dialogue Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data/clips")
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path("src/static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)


class ProcessRequest(BaseModel):
    url: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    name: Optional[str] = None
    language: Optional[str] = "Telugu"
    dialogue_script: Optional[str] = None


@app.post("/api/process")
def process_clip(req: ProcessRequest):
    """Downloads a YouTube segment and separates vocals from BGM."""
    try:
        logger.info(f"Processing URL: {req.url} ({req.start_time} -> {req.end_time})")
        info = download_youtube_clip(
            url=req.url,
            output_dir=DATA_DIR,
            start_time=req.start_time,
            end_time=req.end_time,
            clip_id=req.name
        )
        clip_dir = Path(info["video_path"]).parent

        info_path = clip_dir / "info.json"
        if req.language or req.dialogue_script:
            with open(info_path, "r", encoding="utf-8") as f:
                current_info = json.load(f)
            current_info["language"] = req.language
            current_info["dialogue_script"] = req.dialogue_script
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(current_info, f, indent=2)

        result = separate_audio_stems(clip_dir=clip_dir)
        return {
            "status": "success",
            "clip_id": clip_dir.name,
            "data": result
        }
    except Exception as e:
        logger.exception("Failed to process clip")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clips")
def list_clips():
    """Returns all processed clips in the library."""
    clips = []
    for clip_folder in DATA_DIR.iterdir():
        if clip_folder.is_dir():
            info_file = clip_folder / "info.json"
            if info_file.exists():
                try:
                    with open(info_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["id"] = clip_folder.name
                        data["has_stems"] = (clip_folder / "vocals.wav").exists() and (clip_folder / "instrumental.wav").exists()
                        data["has_final_dub"] = (clip_folder / "video_dubbed_final.mp4").exists()
                        clips.append(data)
                except Exception as e:
                    logger.warning(f"Failed to read {info_file}: {e}")
    return {"clips": clips}


@app.get("/api/clips/{clip_id}")
def get_clip(clip_id: str):
    """Retrieves metadata for a specific clip."""
    clip_folder = DATA_DIR / clip_id
    info_file = clip_folder / "info.json"
    if not info_file.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    with open(info_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        data["id"] = clip_folder.name
        data["has_stems"] = (clip_folder / "vocals.wav").exists() and (clip_folder / "instrumental.wav").exists()
        data["has_final_dub"] = (clip_folder / "video_dubbed_final.mp4").exists()
        return data


@app.get("/api/clips/{clip_id}/segments")
def get_segments(clip_id: str):
    """Retrieves or auto-detects dialogue segments with character voice assignments."""
    clip_folder = DATA_DIR / clip_id
    if not clip_folder.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    try:
        segments = get_or_create_segments(clip_folder)
        characters = get_characters(clip_folder)
        return {"clip_id": clip_id, "segments": segments, "characters": characters}
    except Exception as e:
        logger.exception("Failed to get segments")
        raise HTTPException(status_code=500, detail=str(e))


class SaveSegmentsRequest(BaseModel):
    segments: List[Dict[str, Any]]


@app.post("/api/clips/{clip_id}/segments")
def update_segments(clip_id: str, req: SaveSegmentsRequest):
    """Saves user-trimmed, edited, or re-assigned character segments."""
    clip_folder = DATA_DIR / clip_id
    if not clip_folder.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    save_segments(clip_folder, req.segments)
    return {"status": "success", "segments": req.segments}


@app.get("/api/clips/{clip_id}/characters")
def list_characters(clip_id: str):
    """Returns the list of character roles / players for a clip."""
    clip_folder = DATA_DIR / clip_id
    if not clip_folder.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    characters = get_characters(clip_folder)
    return {"clip_id": clip_id, "characters": characters}


class CharacterActionRequest(BaseModel):
    action: str  # "add" or "update"
    id: Optional[str] = None
    name: str
    color: Optional[str] = None


@app.post("/api/clips/{clip_id}/characters")
def manage_character(clip_id: str, req: CharacterActionRequest):
    """Adds a new player or updates/renames an existing character."""
    clip_folder = DATA_DIR / clip_id
    if not clip_folder.exists():
        raise HTTPException(status_code=404, detail="Clip not found")

    if req.action == "add":
        new_char = add_new_character(clip_folder, name=req.name, color=req.color)
        chars = get_characters(clip_folder)
        return {"status": "success", "characters": chars, "new_character": new_char}
    elif req.action == "update":
        if not req.id:
            raise HTTPException(status_code=400, detail="Character id required for update")
        chars = update_character(clip_folder, char_id=req.id, name=req.name, color=req.color)
        return {"status": "success", "characters": chars}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")


class DiarizeRequest(BaseModel):
    n_speakers: Optional[int] = None


@app.post("/api/clips/{clip_id}/diarize")
def re_cluster_speakers(clip_id: str, req: DiarizeRequest):
    """Re-runs local acoustic voice clustering with optional number of speakers."""
    clip_folder = DATA_DIR / clip_id
    if not clip_folder.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    try:
        res = cluster_speakers_for_clip(clip_folder, n_speakers=req.n_speakers)
        return res
    except Exception as e:
        logger.exception("Failed to cluster speakers")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clips/{clip_id}/dub-chunk")
async def upload_dub_chunk(
    clip_id: str,
    chunk_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Saves user's recorded audio, strips silence, and stores clean WAV."""
    clip_folder = DATA_DIR / clip_id
    if not clip_folder.exists():
        raise HTTPException(status_code=404, detail="Clip not found")

    dubs_dir = clip_folder / "dubs"
    dubs_dir.mkdir(parents=True, exist_ok=True)

    raw_path = dubs_dir / f"{chunk_id}_raw.webm"
    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    wav_path = dubs_dir / f"{chunk_id}.wav"
    trim_info = process_and_trim_chunk(raw_path, wav_path)

    # Keep webm copy
    webm_path = dubs_dir / f"{chunk_id}.webm"
    shutil.copy2(raw_path, webm_path)

    segments_file = clip_folder / "segments.json"
    if segments_file.exists():
        with open(segments_file, "r", encoding="utf-8") as f:
            segments = json.load(f)
        for seg in segments:
            if seg["id"] == chunk_id:
                seg["dubbed"] = True
                seg["dub_duration"] = round(trim_info["trimmed_duration"], 2)
                break
        with open(segments_file, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)

    return {
        "status": "success",
        "chunk_id": chunk_id,
        "file_path": str(wav_path),
        "audio_url": f"/media/{clip_id}/dubs/{chunk_id}.wav",
        "trim_info": trim_info
    }


@app.post("/api/clips/{clip_id}/assemble")
def assemble_clip_dub(clip_id: str):
    """Assembles all dubbed chunks onto the movie instrumental track and video."""
    clip_folder = DATA_DIR / clip_id
    if not clip_folder.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    try:
        result = assemble_full_dub(clip_folder)
        result["video_url"] = f"/media/{clip_id}/video_dubbed_final.mp4"
        result["audio_url"] = f"/media/{clip_id}/audio_dubbed_final.wav"
        return result
    except Exception as e:
        logger.exception("Failed to assemble dub")
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/media", StaticFiles(directory=str(DATA_DIR)), name="media")


@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
