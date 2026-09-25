# 🎙️ Dubalike — Voice Dubbing & Dialogue Studio for Indian Cinema

An AI-powered video dialogue extractor, stem separator, and real-time voice-dubbing studio tailored for Telugu and Hindi cinema.

---

## 🌟 What This Does

1. **Precision YouTube Dialogue Slicing**:
   - Takes any YouTube URL and extracts a short scene (e.g. `00:15` to `00:28`) using `yt-dlp` and `ffmpeg` without having to download entire 2-hour movies.
2. **AI Stem Separation (Meta Demucs)**:
   - Uses Meta's state-of-the-art `htdemucs` neural network (accelerated on Apple Silicon via PyTorch MPS / CPU).
   - Isolates the dialogue (`vocals.wav`) from the background music and sound effects (`instrumental.wav`).
   - Automatically composites a **dub-ready video** (`video_instrumental.mp4`) that keeps the movie's original background score and action sound effects while muting the actor's voice!
3. **Interactive Multi-Track Studio & Live Dubbing**:
   - Web previewer with dynamic track solo/mute toggles (**Full Mix**, **Dialogue Only**, **BGM Only**).
   - Real-time dual volume sliders to inspect the quality of the vocal isolation.
   - **Microphone Dubbing Engine**: 3-second countdown timer, teleprompter/karaoke dialogue cues, live mic recording over the original BGM, and instant playback!

---

## 🚀 Quick Start

### 1. Prerequisites
- macOS or Linux with `ffmpeg` installed (`brew install ffmpeg`)
- Python 3.11 (`brew install python@3.11`)

### 2. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 3. Launch the Web Studio
```bash
uvicorn src.server:app --reload --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser!

---

## 💻 Command Line Interface (CLI)

You can also process YouTube clips directly from your terminal:

```bash
# Example: Extract a 12-second dialogue
python -m src.cli \
  --url "https://www.youtube.com/watch?v=kwtYF_iY8rY" \
  --start "00:10" \
  --end "00:22" \
  --name "pushpa_thaggedhe_le"
```

### Generated Files per Clip (`data/clips/<clip_id>/`):
- `video.mp4` — Original video clip with full audio
- `vocals.wav` — Clean isolated actor dialogue (speech only)
- `instrumental.wav` — Clean isolated BGM + sound effects
- `video_instrumental.mp4` — Dub-ready video (original visuals + pure BGM/SFX)
- `video_vocals.mp4` — Rehearsal video (original visuals + isolated dialogue)
- `info.json` — Metadata, timestamps, and script prompts

---

## 🛠️ Tech Stack
- **AI Separation**: Meta Demucs (`htdemucs`), PyTorch, TorchAudio
- **Video & Audio Slicing**: `yt-dlp`, FFmpeg
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Vanilla HTML5, Web Audio API, MediaRecorder API
