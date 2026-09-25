import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def separate_audio_stems(
    clip_dir: Path,
    model_name: str = "htdemucs",
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Separates speech (vocals) from background music/SFX (instrumental) using Meta's Demucs.
    Outputs:
    - vocals.wav: pure speech/dialogue track
    - instrumental.wav: pure background score + sound effects
    - video_instrumental.mp4: original video paired with isolated BGM (perfect for dubbing)
    - video_vocals.mp4: original video paired with isolated vocals (perfect for rehearsal)
    """
    clip_dir = Path(clip_dir)
    audio_path = clip_dir / "audio.wav"
    video_path = clip_dir / "video.mp4"
    info_path = clip_dir / "info.json"

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found at {audio_path}")

    logger.info(f"Starting stem separation for {audio_path} using model '{model_name}'...")

    temp_demucs_out = clip_dir / "_demucs_temp"
    temp_demucs_out.mkdir(parents=True, exist_ok=True)

    # Resolve python binary inside .venv or sys.executable
    python_bin = sys.executable

    cmd = [
        python_bin, "-m", "demucs",
        "--two-stems=vocals",
        "-n", model_name,
        "-o", str(temp_demucs_out),
    ]

    if device:
        cmd.extend(["-d", device])

    cmd.append(str(audio_path))

    logger.info(f"Running separation command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Demucs separation failed: {result.stderr}")
        raise RuntimeError(f"Demucs separation failed: {result.stderr}")

    # Demucs places files in: temp_demucs_out / model_name / audio_stem_name / [vocals.wav, no_vocals.wav]
    stem_folder = temp_demucs_out / model_name / audio_path.stem
    if not stem_folder.exists():
        # Search recursively for vocals.wav in temp directory
        found_vocals = list(temp_demucs_out.rglob("vocals.wav"))
        if found_vocals:
            stem_folder = found_vocals[0].parent
        else:
            raise FileNotFoundError(f"Could not locate separated stems in {temp_demucs_out}")

    dest_vocals = clip_dir / "vocals.wav"
    dest_instrumental = clip_dir / "instrumental.wav"

    src_vocals = stem_folder / "vocals.wav"
    src_instrumental = stem_folder / "no_vocals.wav"

    if not src_vocals.exists() or not src_instrumental.exists():
        raise FileNotFoundError(f"Missing vocals or no_vocals stem in {stem_folder}")

    shutil.move(str(src_vocals), str(dest_vocals))
    shutil.move(str(src_instrumental), str(dest_instrumental))

    # Clean up temp demucs dir
    shutil.rmtree(str(temp_demucs_out), ignore_errors=True)
    logger.info("Stems extracted: vocals.wav and instrumental.wav")

    # Composite videos if original video exists
    video_instrumental_path = clip_dir / "video_instrumental.mp4"
    video_vocals_path = clip_dir / "video_vocals.mp4"

    if video_path.exists():
        logger.info("Generating dub-ready video (video + isolated BGM)...")
        ffmpeg_bgm = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(dest_instrumental),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(video_instrumental_path)
        ]
        subprocess.run(ffmpeg_bgm, capture_output=True, check=True)

        logger.info("Generating rehearsal video (video + isolated dialogue)...")
        ffmpeg_voc = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(dest_vocals),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(video_vocals_path)
        ]
        subprocess.run(ffmpeg_voc, capture_output=True, check=True)

    # Update metadata
    info_data = {}
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info_data = json.load(f)

    info_data.update({
        "separated": True,
        "vocals_path": str(dest_vocals),
        "instrumental_path": str(dest_instrumental),
        "video_instrumental_path": str(video_instrumental_path) if video_instrumental_path.exists() else None,
        "video_vocals_path": str(video_vocals_path) if video_vocals_path.exists() else None,
    })

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_data, f, indent=2)

    logger.info(f"Stem separation complete for {clip_dir.name}!")
    return info_data
