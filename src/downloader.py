import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_ytdlp_cmd() -> list[str]:
    """Finds the best available yt-dlp binary and configures JS runtimes and client fallbacks."""
    venv_ytdlp = Path(sys.prefix) / "bin" / "yt-dlp"
    bin_path = str(venv_ytdlp) if venv_ytdlp.exists() else shutil.which("yt-dlp") or "yt-dlp"

    # Use android,web client to bypass YouTube web bot block
    cmd = [
        bin_path,
        "--extractor-args", "youtube:player_client=android,web",
    ]

    # Check for node js runtime
    node_path = "/opt/homebrew/opt/node@22/bin/node"
    if os.path.exists(node_path):
        cmd.extend(["--js-runtimes", f"node:{node_path}"])
    elif shutil.which("node"):
        cmd.extend(["--js-runtimes", "node"])

    return cmd


def parse_timestamp_to_seconds(ts: str) -> float:
    """Parses timestamps like '01:23', '00:01:23', '83', or '83.5' into float seconds."""
    ts = str(ts).strip()
    if not ts:
        return 0.0
    parts = ts.split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Invalid timestamp format: {ts}")


def format_seconds_to_hms(seconds: float) -> str:
    """Formats float seconds into HH:MM:SS format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


def fetch_youtube_metadata(url: str) -> Dict[str, Any]:
    """Fetches video metadata (title, duration, thumbnail, id) without downloading the media."""
    cmd = get_ytdlp_cmd() + [
        "--dump-single-json",
        "--no-playlist",
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err_msg = result.stderr.strip() or "Unknown error"
        # Extract clean error message from yt-dlp output
        match = re.search(r"ERROR:\s*(\[youtube\].*)", err_msg)
        if match:
            clean_err = match.group(1)
        else:
            clean_err = err_msg.splitlines()[-1] if err_msg else "Could not fetch YouTube video."
        logger.error(f"yt-dlp metadata fetch failed: {clean_err}")
        raise ValueError(f"YouTube Error: {clean_err}")

    info = json.loads(result.stdout)
    return {
        "id": info.get("id", "clip"),
        "title": info.get("title", "Untitled Clip"),
        "uploader": info.get("uploader", "Unknown"),
        "duration": info.get("duration", 0),
        "thumbnail": info.get("thumbnail", ""),
        "webpage_url": info.get("webpage_url", url)
    }


def download_youtube_clip(
    url: str,
    output_dir: Path,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    clip_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Downloads a YouTube clip (or timestamped segment) and extracts:
    - video.mp4 (video with audio)
    - audio.wav (uncompressed 44.1kHz stereo audio for AI separation)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching metadata for {url}...")
    meta = fetch_youtube_metadata(url)
    clean_id = clip_id or re.sub(r"[^\w\-]", "_", meta["id"])
    target_dir = output_dir / clean_id
    target_dir.mkdir(parents=True, exist_ok=True)

    video_path = target_dir / "video.mp4"
    audio_path = target_dir / "audio.wav"

    # Build yt-dlp command
    cmd = get_ytdlp_cmd() + [
        "--no-playlist",
        "--force-overwrites",
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "--merge-output-format", "mp4",
    ]

    has_section = False
    if start_time is not None or end_time is not None:
        start_sec = parse_timestamp_to_seconds(start_time) if start_time else 0.0
        end_sec = parse_timestamp_to_seconds(end_time) if end_time else float(meta["duration"])
        
        start_hms = format_seconds_to_hms(start_sec)
        end_hms = format_seconds_to_hms(end_sec)
        section_arg = f"*{start_hms}-{end_hms}"
        cmd.extend([
            "--download-sections", section_arg,
            "--force-keyframes-at-cuts"
        ])
        has_section = True
        logger.info(f"Downloading section: {section_arg} ({end_sec - start_sec:.2f} seconds)")

    cmd.extend([
        "-o", str(video_path),
        url
    ])

    logger.info(f"Running download command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err_msg = result.stderr.strip() or "Unknown error"
        match = re.search(r"ERROR:\s*(\[youtube\].*)", err_msg)
        clean_err = match.group(1) if match else (err_msg.splitlines()[-1] if err_msg else "Download failed.")
        logger.error(f"yt-dlp failed: {clean_err}")
        raise RuntimeError(f"YouTube Download Error: {clean_err}")

    if not video_path.exists():
        # Sometimes yt-dlp appends .mp4 or changes extension
        candidates = list(target_dir.glob("video*"))
        if candidates:
            candidates[0].rename(video_path)
        else:
            raise FileNotFoundError(f"Failed to locate downloaded video at {video_path}")

    # Extract clean, loudness-normalized 44.1kHz 16-bit stereo WAV using ffmpeg loudnorm
    logger.info(f"Extracting normalized audio to {audio_path}...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,aformat=sample_rates=44100",
        "-acodec", "pcm_s16le",
        "-ac", "2",
        str(audio_path)
    ]
    ffmpeg_res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if ffmpeg_res.returncode != 0:
        logger.error(f"ffmpeg audio extraction failed: {ffmpeg_res.stderr}")
        raise RuntimeError(f"ffmpeg audio extraction failed: {ffmpeg_res.stderr}")

    # Remux normalized audio back into video.mp4 so original video audio is loud & clear
    temp_norm_vid = target_dir / "video_norm.mp4"
    remux_cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        str(temp_norm_vid)
    ]
    remux_res = subprocess.run(remux_cmd, capture_output=True, text=True)
    if remux_res.returncode == 0 and temp_norm_vid.exists():
        temp_norm_vid.replace(video_path)

    info_data = {
        "clip_id": clean_id,
        "title": meta["title"],
        "uploader": meta["uploader"],
        "url": url,
        "start_time": start_time,
        "end_time": end_time,
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "thumbnail": meta["thumbnail"],
        "duration": meta["duration"]
    }

    with open(target_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info_data, f, indent=2)

    logger.info(f"Successfully downloaded and extracted audio for {clean_id}!")
    return info_data
