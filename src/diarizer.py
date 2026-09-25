import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHARACTER_PALETTE = [
    {"id": "char_1", "name": "Character 1", "color": "#3b82f6"},  # Blue
    {"id": "char_2", "name": "Character 2", "color": "#a855f7"},  # Purple
    {"id": "char_3", "name": "Character 3", "color": "#10b981"},  # Emerald Green
    {"id": "char_4", "name": "Character 4", "color": "#f59e0b"},  # Amber / Gold
    {"id": "char_5", "name": "Character 5", "color": "#ec4899"},  # Pink / Rose
    {"id": "char_6", "name": "Character 6", "color": "#06b6d4"},  # Cyan
    {"id": "char_7", "name": "Character 7", "color": "#f97316"},  # Orange
    {"id": "char_8", "name": "Character 8", "color": "#6366f1"},  # Indigo
]


def get_characters(clip_dir: Path) -> List[Dict[str, Any]]:
    """Loads existing characters.json or returns default 2 characters."""
    clip_dir = Path(clip_dir)
    char_file = clip_dir / "characters.json"
    if char_file.exists():
        try:
            with open(char_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {char_file}: {e}")

    # Default to 2 characters
    default_chars = DEFAULT_CHARACTER_PALETTE[:2]
    save_characters(clip_dir, default_chars)
    return default_chars


def save_characters(clip_dir: Path, characters: List[Dict[str, Any]]) -> None:
    """Saves characters list to characters.json."""
    clip_dir = Path(clip_dir)
    char_file = clip_dir / "characters.json"
    with open(char_file, "w", encoding="utf-8") as f:
        json.dump(characters, f, indent=2)


def add_new_character(clip_dir: Path, name: str, color: Optional[str] = None) -> Dict[str, Any]:
    """Adds a new character / player to the clip."""
    characters = get_characters(clip_dir)
    idx = len(characters) + 1
    new_id = f"char_{idx}"

    if not color:
        color = DEFAULT_CHARACTER_PALETTE[(idx - 1) % len(DEFAULT_CHARACTER_PALETTE)]["color"]

    new_char = {"id": new_id, "name": name or f"Character {idx}", "color": color}
    characters.append(new_char)
    save_characters(clip_dir, characters)
    return new_char


def update_character(clip_dir: Path, char_id: str, name: str, color: Optional[str] = None) -> List[Dict[str, Any]]:
    """Renames or updates a character."""
    characters = get_characters(clip_dir)
    for c in characters:
        if c["id"] == char_id:
            c["name"] = name
            if color:
                c["color"] = color
            break
    save_characters(clip_dir, characters)

    # Sync name in segments.json if it exists
    seg_file = clip_dir / "segments.json"
    if seg_file.exists():
        try:
            with open(seg_file, "r", encoding="utf-8") as f:
                segments = json.load(f)
            updated = False
            for s in segments:
                if s.get("character_id") == char_id:
                    s["character_name"] = name
                    updated = True
            if updated:
                with open(seg_file, "w", encoding="utf-8") as f:
                    json.dump(segments, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not sync segments with character rename: {e}")

    return characters


def extract_voice_feature(audio_chunk: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extracts acoustic fingerprint: MFCCs (16 coefficients: mean + std) + Spectral Centroid + Pitch.
    Total: 35 features per chunk.
    """
    if len(audio_chunk) < 800:
        audio_chunk = np.pad(audio_chunk, (0, 800 - len(audio_chunk)))

    t_audio = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0)
    target_sr = 16000
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        t_audio = resampler(t_audio)
    else:
        target_sr = sr

    # 1. MFCC
    mfcc_fn = T.MFCC(sample_rate=target_sr, n_mfcc=16, melkwargs={"n_fft": 512, "n_mels": 48})
    mfccs = mfcc_fn(t_audio).squeeze(0)  # [16, T]
    mfcc_mean = mfccs.mean(dim=1).numpy()
    mfcc_std = mfccs.std(dim=1).numpy()

    # 2. Spectral Centroid
    cent_fn = T.SpectralCentroid(sample_rate=target_sr, n_fft=512)
    cent = cent_fn(t_audio).squeeze(0)
    cent_mean = cent.mean().item()
    cent_std = cent.std().item()

    # 3. Autocorrelation Pitch Estimation in human speech range (80-400Hz)
    np_chunk = t_audio.squeeze(0).numpy()
    corr = np.correlate(np_chunk, np_chunk, mode="full")[len(np_chunk) - 1 :]
    # Search lags: 16000/400=40 to 16000/80=200
    search_window = corr[40:200]
    peak_lag = 40 + np.argmax(search_window) if len(search_window) > 0 else 100
    est_pitch = float(target_sr / peak_lag)

    return np.concatenate([mfcc_mean, mfcc_std, [cent_mean, cent_std, est_pitch]])


def cluster_speakers_for_clip(clip_dir: Path, n_speakers: Optional[int] = None) -> Dict[str, Any]:
    """
    Performs offline acoustic clustering on all dialogue chunks in a clip.
    Assigns each chunk a character_id and character_name.
    """
    clip_dir = Path(clip_dir)
    vocals_file = clip_dir / "vocals.wav"
    segments_file = clip_dir / "segments.json"

    if not vocals_file.exists():
        raise FileNotFoundError(f"Vocals track not found at {vocals_file}")
    if not segments_file.exists():
        raise FileNotFoundError(f"Segments file not found at {segments_file}")

    with open(segments_file, "r", encoding="utf-8") as f:
        segments: List[Dict[str, Any]] = json.load(f)

    if not segments:
        return {"status": "no_segments", "segments": []}

    audio, sr = sf.read(str(vocals_file))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    existing_characters = get_characters(clip_dir)
    char_map = {c["id"]: c for c in existing_characters}

    if len(segments) == 1:
        char_1 = existing_characters[0]
        segments[0]["character_id"] = char_1["id"]
        segments[0]["character_name"] = char_1["name"]
        with open(segments_file, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)
        return {"status": "success", "segments": segments, "characters": existing_characters}

    # Extract acoustic feature vector for each segment
    features = []
    for seg in segments:
        s = max(0, int(seg["start"] * sr))
        e = min(len(audio), int(seg["end"] * sr))
        chunk_data = audio[s:e]
        feat = extract_voice_feature(chunk_data, sr=sr)
        features.append(feat)

    features = np.array(features)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Determine number of clusters
    k = n_speakers
    if not k or k < 1:
        if len(segments) == 2:
            k = 2
        else:
            # Check silhouette score for k in [2, min(3, len(segments))]
            max_k = min(3, len(segments))
            best_k = 2
            best_score = -1.0
            for cand_k in range(2, max_k + 1):
                try:
                    labels = AgglomerativeClustering(n_clusters=cand_k).fit(X).labels_
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_k = cand_k
                except Exception:
                    pass
            k = best_k

    k = min(k, len(segments))

    # Ensure we have at least k characters in characters.json
    while len(existing_characters) < k:
        idx = len(existing_characters) + 1
        palette_entry = DEFAULT_CHARACTER_PALETTE[(idx - 1) % len(DEFAULT_CHARACTER_PALETTE)]
        new_char = {"id": f"char_{idx}", "name": f"Character {idx}", "color": palette_entry["color"]}
        existing_characters.append(new_char)

    save_characters(clip_dir, existing_characters)
    char_map = {c["id"]: c for c in existing_characters}

    clustering = AgglomerativeClustering(n_clusters=k).fit(X)
    labels = clustering.labels_

    for idx, seg in enumerate(segments):
        assigned_char_id = f"char_{labels[idx] + 1}"
        char_obj = char_map.get(assigned_char_id, existing_characters[0])
        seg["character_id"] = char_obj["id"]
        seg["character_name"] = char_obj["name"]

    with open(segments_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)

    logger.info(f"Clustered {len(segments)} dialogue chunks into {k} characters for {clip_dir.name}")
    return {
        "status": "success",
        "clip_id": clip_dir.name,
        "clusters_count": k,
        "segments": segments,
        "characters": existing_characters
    }
