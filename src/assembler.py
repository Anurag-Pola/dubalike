import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple
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


def trim_silence_from_audio(
    audio_data: np.ndarray,
    sr: int = 44100,
    pad_sec: float = 0.06,
    energy_thresh_ratio: float = 0.07,
    min_energy_thresh: float = 0.012
) -> Tuple[np.ndarray, float, float]:
    """
    Detects speech onset and offset via RMS energy envelope and strips leading & trailing dead air.
    Returns (trimmed_audio_data, start_offset_seconds, end_offset_seconds).
    """
    if audio_data.size == 0:
        return audio_data, 0.0, 0.0

    # Calculate mono envelope
    mono = np.mean(np.abs(audio_data), axis=1) if audio_data.ndim > 1 else np.abs(audio_data)
    peak = np.max(mono)
    total_duration = len(audio_data) / sr

    if peak < 0.005:
        # Audio is completely silent
        return audio_data, 0.0, total_duration

    frame_len = int(sr * 0.02)  # 20ms frame
    n_frames = len(mono) // frame_len
    if n_frames == 0:
        return audio_data, 0.0, total_duration

    reshaped = mono[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(reshaped ** 2, axis=1))
    peak_rms = np.max(rms)

    thresh = max(min_energy_thresh, peak_rms * energy_thresh_ratio)
    speech_indices = np.where(rms > thresh)[0]

    if len(speech_indices) == 0:
        return audio_data, 0.0, total_duration

    first_idx = speech_indices[0]
    last_idx = speech_indices[-1]
    pad_samples = int(pad_sec * sr)

    start_samp = max(0, first_idx * frame_len - pad_samples)
    end_samp = min(len(audio_data), (last_idx + 1) * frame_len + pad_samples)

    trimmed = audio_data[start_samp:end_samp].copy()

    # Subtle 5ms fade-in and fade-out to prevent clicks
    fade_len = min(int(0.005 * sr), len(trimmed) // 4)
    if fade_len > 0:
        fade_in = np.linspace(0.0, 1.0, fade_len)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        if trimmed.ndim == 2:
            trimmed[:fade_len] *= fade_in[:, None]
            trimmed[-fade_len:] *= fade_out[:, None]
        else:
            trimmed[:fade_len] *= fade_in
            trimmed[-fade_len:] *= fade_out

    start_sec = start_samp / sr
    end_sec = end_samp / sr
    return trimmed, start_sec, end_sec


def process_and_trim_chunk(
    input_path: Path,
    output_wav_path: Path,
    sample_rate: int = 44100
) -> Dict[str, Any]:
    """
    Takes an uploaded dub (e.g. webm), converts it to WAV, strips leading and trailing silence,
    and writes out the clean WAV file.
    """
    temp_wav = output_wav_path.parent / f"temp_{output_wav_path.name}"
    convert_to_wav(input_path, temp_wav, sample_rate=sample_rate)

    audio_data, sr = sf.read(str(temp_wav))
    if audio_data.ndim == 1:
        audio_data = np.stack([audio_data, audio_data], axis=1)

    orig_dur = len(audio_data) / sr
    trimmed_data, lead_cut, trail_cut = trim_silence_from_audio(audio_data, sr=sr)
    trimmed_dur = len(trimmed_data) / sr

    sf.write(str(output_wav_path), trimmed_data, sr)
    temp_wav.unlink(missing_ok=True)

    logger.info(
        f"Processed dub chunk: original={orig_dur:.2f}s -> trimmed={trimmed_dur:.2f}s "
        f"(trimmed {lead_cut:.2f}s lead, {orig_dur - trail_cut:.2f}s trail)"
    )

    return {
        "original_duration": orig_dur,
        "trimmed_duration": trimmed_dur,
        "lead_silence_stripped": lead_cut,
        "wav_path": str(output_wav_path)
    }


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
        chunk_wav = dubs_dir / f"{chunk_id}.wav"
        chunk_webm = dubs_dir / f"{chunk_id}.webm"

        target_chunk_audio = None
        if chunk_wav.exists():
            target_chunk_audio = chunk_wav
        elif chunk_webm.exists():
            # Process & trim webm to wav
            process_and_trim_chunk(chunk_webm, chunk_wav, sample_rate=sr)
            target_chunk_audio = chunk_wav

        if target_chunk_audio and target_chunk_audio.exists():
            chunk_data, chunk_sr = sf.read(str(target_chunk_audio))
            if chunk_sr != sr:
                temp_resampled = dubs_dir / f"{chunk_id}_resampled.wav"
                convert_to_wav(target_chunk_audio, temp_resampled, sample_rate=sr)
                chunk_data, _ = sf.read(str(temp_resampled))
                temp_resampled.unlink(missing_ok=True)

            if chunk_data.ndim == 1:
                chunk_data = np.stack([chunk_data, chunk_data], axis=1)

            # Optional pass of silence trimming if not already trimmed
            chunk_trimmed, _, _ = trim_silence_from_audio(chunk_data, sr=sr)
            if len(chunk_trimmed) > 0:
                chunk_data = chunk_trimmed

            start_sample = int(seg["start"] * sr)
            end_sample = min(total_samples, start_sample + len(chunk_data))
            chunk_len = end_sample - start_sample

            if chunk_len > 0:
                # Add chunk to dub track with slight gain boost for vocal clarity
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
