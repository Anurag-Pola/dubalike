import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def detect_speech_segments(
    vocals_path: Path,
    min_speech_sec: float = 0.5,
    min_silence_sec: float = 0.35,
    padding_sec: float = 0.12
) -> List[Dict[str, Any]]:
    """
    Detects continuous speech / dialogue chunks from vocals.wav using RMS energy envelope.
    """
    vocals_path = Path(vocals_path)
    if not vocals_path.exists():
        raise FileNotFoundError(f"Vocals file not found at {vocals_path}")

    audio, sr = sf.read(str(vocals_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    total_duration = len(audio) / sr
    if total_duration <= 0.2:
        return []

    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.010)
    num_frames = (len(audio) - frame_len) // hop_len

    if num_frames <= 0:
        return [{"id": "chunk_0", "start": 0.0, "end": round(total_duration, 2), "duration": round(total_duration, 2), "text": "", "dubbed": False}]

    frames = np.lib.stride_tricks.sliding_window_view(audio[:num_frames * hop_len + frame_len], frame_len)[::hop_len]
    rms = np.sqrt(np.mean(frames**2, axis=1))

    percentile_95 = float(np.percentile(rms, 95))
    thresh = max(0.012, percentile_95 * 0.12)
    is_speech = rms > thresh

    raw_segments = []
    in_speech = False
    start_frame = 0

    for i, active in enumerate(is_speech):
        if active and not in_speech:
            in_speech = True
            start_frame = i
        elif not active and in_speech:
            in_speech = False
            duration = (i - start_frame) * hop_len / sr
            if duration >= min_speech_sec:
                raw_segments.append([start_frame * hop_len / sr, i * hop_len / sr])

    if in_speech:
        duration = (len(is_speech) - start_frame) * hop_len / sr
        if duration >= min_speech_sec:
            raw_segments.append([start_frame * hop_len / sr, len(is_speech) * hop_len / sr])

    if not raw_segments:
        return [{"id": "chunk_0", "start": 0.0, "end": round(total_duration, 2), "duration": round(total_duration, 2), "text": "", "dubbed": False}]

    merged = []
    for s_start, s_end in raw_segments:
        if not merged:
            merged.append([s_start, s_end])
        else:
            prev_start, prev_end = merged[-1]
            if s_start - prev_end <= min_silence_sec:
                merged[-1][1] = s_end
            else:
                merged.append([s_start, s_end])

    chunks = []
    for idx, (s, e) in enumerate(merged):
        padded_start = max(0.0, s - padding_sec)
        padded_end = min(total_duration, e + padding_sec)
        dur = round(padded_end - padded_start, 2)
        chunks.append({
            "id": f"chunk_{idx}",
            "start": round(padded_start, 2),
            "end": round(padded_end, 2),
            "duration": dur,
            "text": "",
            "dubbed": False
        })

    return chunks


def get_or_create_segments(clip_dir: Path) -> List[Dict[str, Any]]:
    """Retrieves existing segments.json or auto-generates them and performs speaker clustering."""
    clip_dir = Path(clip_dir)
    segments_file = clip_dir / "segments.json"
    vocals_file = clip_dir / "vocals.wav"

    if segments_file.exists():
        try:
            with open(segments_file, "r", encoding="utf-8") as f:
                segments = json.load(f)
                # Check if segments have character_id assigned, if not, cluster
                if segments and ("character_id" not in segments[0]):
                    from src.diarizer import cluster_speakers_for_clip
                    try:
                        res = cluster_speakers_for_clip(clip_dir)
                        return res.get("segments", segments)
                    except Exception as e:
                        logger.warning(f"Failed to cluster speakers for existing segments: {e}")
                return segments
        except Exception as e:
            logger.warning(f"Failed to read {segments_file}: {e}")

    if not vocals_file.exists():
        return []

    chunks = detect_speech_segments(vocals_file)

    info_file = clip_dir / "info.json"
    if info_file.exists():
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                info = json.load(f)
                if info.get("dialogue_script") and chunks:
                    chunks[0]["text"] = info["dialogue_script"]
        except Exception:
            pass

    save_segments(clip_dir, chunks)

    # Perform speaker clustering
    from src.diarizer import cluster_speakers_for_clip
    try:
        clustered = cluster_speakers_for_clip(clip_dir)
        return clustered.get("segments", chunks)
    except Exception as e:
        logger.warning(f"Could not cluster speakers during initial creation: {e}")
        return chunks


def save_segments(clip_dir: Path, segments: List[Dict[str, Any]]) -> None:
    """Saves segments list to segments.json."""
    clip_dir = Path(clip_dir)
    segments_file = clip_dir / "segments.json"
    with open(segments_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)
    logger.info(f"Saved {len(segments)} segments to {segments_file}")
