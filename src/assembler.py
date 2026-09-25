import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def convert_to_wav(input_path: Path, output_wav_path: Path, sample_rate: int = 44100) -> None:
    """Converts any audio file (e.g. webm from MediaRecorder) to uncompressed WAV."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "2",
        str(output_wav_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def assemble_full_dub(clip_dir: Path) -> Dict[str, Any]:
    """
    Overlays all recorded chunk audios onto the movie's instrumental track at exact timestamps,
    and renders video_dubbed_final.mp4.
    """
    clip_dir = Path(clip_dir)
    inst_path = clip_dir / "instrumental.wav"
    video_path = clip_dir / "video.mp4"
    segments_path = clip_dir / "segments.json"
    dubs_dir = clip_dir / "dubs"

    if not inst_path.exists():
        raise FileNotFoundError(f"Instrumental track not found at {inst_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at {video_path}")
    if not segments_path.exists():
        raise FileNotFoundError(f"Segments metadata not found at {segments_path}")

    with open(segments_path, "r", encoding="utf-8") as f:
        segments: List[Dict[str, Any]] = json.load(f)

    # Load base instrumental track
    inst_audio, sr = sf.read(str(inst_path))
    if inst_audio.ndim == 1:
        inst_audio = np.stack([inst_audio, inst_audio], axis=1)

    total_samples = len(inst_audio)
    dub_track = np.zeros_like(inst_audio)

    dubbed_count = 0
    for seg in segments:
        chunk_id = seg["id"]
        # Look for chunk audio in dubs_dir (wav or webm)
        chunk_wav = dubs_dir / f"{chunk_id}.wav"
        chunk_webm = dubs_dir / f"{chunk_id}.webm"

        target_chunk_audio = None
        if chunk_wav.exists():
            target_chunk_audio = chunk_wav
        elif chunk_webm.exists():
            # Convert webm to wav
            convert_to_wav(chunk_webm, chunk_wav, sample_rate=sr)
            target_chunk_audio = chunk_wav

        if target_chunk_audio and target_chunk_audio.exists():
            chunk_data, chunk_sr = sf.read(str(target_chunk_audio))
            if chunk_sr != sr:
                # Resample or convert via ffmpeg
                temp_resampled = dubs_dir / f"{chunk_id}_resampled.wav"
                convert_to_wav(target_chunk_audio, temp_resampled, sample_rate=sr)
                chunk_data, _ = sf.read(str(temp_resampled))
                temp_resampled.unlink(missing_ok=True)

            if chunk_data.ndim == 1:
                chunk_data = np.stack([chunk_data, chunk_data], axis=1)

            start_sample = int(seg["start"] * sr)
            end_sample = min(total_samples, start_sample + len(chunk_data))
            chunk_len = end_sample - start_sample

            if chunk_len > 0:
                # Add chunk to dub track with slight gain boost for vocal prominence
                dub_track[start_sample:end_sample] += chunk_data[:chunk_len] * 1.15
                dubbed_count += 1
                seg["dubbed"] = True

    if dubbed_count == 0:
        raise ValueError("No recorded dub chunks found. Record and accept at least one dialogue chunk first.")

    # Save updated segments
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)

    # Mix instrumental + user dub track
    mixed_audio = inst_audio + dub_track
    # Soft clip to prevent distortion
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0.98:
        mixed_audio = mixed_audio / max_val * 0.95

    out_audio_path = clip_dir / "audio_dubbed_final.wav"
    sf.write(str(out_audio_path), mixed_audio, sr)
    logger.info(f"Assembled composite audio with {dubbed_count} chunks to {out_audio_path}")

    # Multiplex with video
    out_video_path = clip_dir / "video_dubbed_final.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(out_audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        str(out_video_path)
    ]
    subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
    logger.info(f"Assembled final dubbed video: {out_video_path}")

    return {
        "status": "success",
        "clip_id": clip_dir.name,
        "dubbed_chunks_count": dubbed_count,
        "video_dubbed_path": str(out_video_path),
        "audio_dubbed_path": str(out_audio_path)
    }
