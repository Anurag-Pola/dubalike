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

from src.assembler import assemble_full_dub
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
        # Step 1: Download
        info = download_youtube_clip(
            url=req.url,
            output_dir=DATA_DIR,
            start_time=req.start_time,
            end_time=req.end_time,
            clip_id=req.name
        )
        clip_dir = Path(info["video_path"]).parent

        # Add language and dialogue script to info if provided
        info_path = clip_dir / "info.json"
        if req.language or req.dialogue_script:
            with open(info_path, "r", encoding="utf-8") as f:
                current_info = json.load(f)
            current_info["language"] = req.language
            current_info["dialogue_script"] = req.dialogue_script
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(current_info, f, indent=2)

        # Step 2: Separate
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
                        clips.append(data)
                except Exception as e:
                    logger.warning(f"Failed to read {info_file}: {e}")
    # Sort clips by id or creation
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
    """Retrieves or auto-detects dialogue segments for a clip."""
    clip_folder = DATA_DIR / clip_id
    if not clip_folder.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    try:
        segments = get_or_create_segments(clip_folder)
        return {"clip_id": clip_id, "segments": segments}
    except Exception as e:
        logger.exception("Failed to get segments")
        raise HTTPException(status_code=500, detail=str(e))


class SaveSegmentsRequest(BaseModel):
    segments: List[Dict[str, Any]]


@app.post("/api/clips/{clip_id}/segments")
def update_segments(clip_id: str, req: SaveSegmentsRequest):
    """Saves user-trimmed or edited segments."""
    clip_folder = DATA_DIR / clip_id
    if not clip_folder.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    save_segments(clip_folder, req.segments)
    return {"status": "success", "segments": req.segments}


@app.post("/api/clips/{clip_id}/dub-chunk")
async def upload_dub_chunk(
    clip_id: str,
    chunk_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Saves a user's recorded audio for a specific dialogue chunk."""
    clip_folder = DATA_DIR / clip_id
    if not clip_folder.exists():
        raise HTTPException(status_code=404, detail="Clip not found")

    dubs_dir = clip_folder / "dubs"
    dubs_dir.mkdir(parents=True, exist_ok=True)

    target_path = dubs_dir / f"{chunk_id}.webm"
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    segments_file = clip_folder / "segments.json"
    if segments_file.exists():
        with open(segments_file, "r", encoding="utf-8") as f:
            segments = json.load(f)
        for seg in segments:
            if seg["id"] == chunk_id:
                seg["dubbed"] = True
                break
        with open(segments_file, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)

    return {"status": "success", "chunk_id": chunk_id, "file_path": str(target_path)}


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


# Mount media folder for streaming
app.mount("/media", StaticFiles(directory=str(DATA_DIR)), name="media")

# Mount static web app
@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
