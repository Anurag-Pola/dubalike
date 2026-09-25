import argparse
import sys
from pathlib import Path

from src.downloader import download_youtube_clip
from src.separator import separate_audio_stems


def main():
    parser = argparse.ArgumentParser(
        description="Download a YouTube clip, cut dialogue segment, and separate Vocals vs Instrumental BGM."
    )
    parser.add_argument("--url", "-u", type=str, required=True, help="YouTube video URL")
    parser.add_argument("--start", "-s", type=str, default=None, help="Start time (e.g. '01:15' or '75')")
    parser.add_argument("--end", "-e", type=str, default=None, help="End time (e.g. '01:28' or '88')")
    parser.add_argument("--name", "-n", type=str, default=None, help="Custom identifier/folder name for the clip")
    parser.add_argument("--output-dir", "-o", type=str, default="data/clips", help="Base output directory")
    parser.add_argument("--model", "-m", type=str, default="htdemucs", help="Demucs model name (default: htdemucs)")
    parser.add_argument("--device", "-d", type=str, default=None, help="Torch device ('cpu', 'mps', or 'cuda')")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🎬 Dubalike — Voice Dubbing & Dialogue Studio")
    print(f"🔗 URL: {args.url}")
    print(f"⏱️  Segment: {args.start or '00:00'} -> {args.end or 'end'}")
    print("=" * 60)

    try:
        # Step 1: Download segment & extract audio
        print("\n[1/2] 📥 Downloading clip and extracting audio...")
        info = download_youtube_clip(
            url=args.url,
            output_dir=output_dir,
            start_time=args.start,
            end_time=args.end,
            clip_id=args.name
        )
        clip_dir = Path(info["video_path"]).parent
        print(f"  ✓ Downloaded to: {clip_dir}")

        # Step 2: Separate vocal dialogue from BGM / SFX
        print("\n[2/2] 🧠 Running AI audio stem separation (Demucs)...")
        result = separate_audio_stems(
            clip_dir=clip_dir,
            model_name=args.model,
            device=args.device
        )

        print("\n" + "=" * 60)
        print("✨ PIPELINE COMPLETE! Files generated:")
        print(f"  🎬 Original Video:      {result.get('video_path', clip_dir / 'video.mp4')}")
        print(f"  🗣️  Isolated Vocals:     {result.get('vocals_path')}")
        print(f"  🎵 Isolated BGM/SFX:    {result.get('instrumental_path')}")
        print(f"  🎤 Dub-Ready Video:     {result.get('video_instrumental_path')}")
        print(f"  🎧 Rehearsal Video:     {result.get('video_vocals_path')}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during processing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
