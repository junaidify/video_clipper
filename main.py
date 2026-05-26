#!/usr/bin/env python3
"""
Video Auto-Clipper
==================
Analyzes video content (via transcript), detects the most compelling
hook-worthy moments, and automatically splits the video into short
vertical clips optimized for TikTok / Reels / YouTube Shorts.

Pipeline:
  1. Transcribe video using OpenAI Whisper (word-level timestamps)
  2. Analyze transcript for hooks: key quotes, interesting points,
     high-value content, emotional peaks
  3. Split video into clips at smart boundaries using FFmpeg
  4. Crop to 9:16 vertical format with fade transitions

Usage:
  python main.py path/to/video.mp4
  python main.py path/to/video.mp4 --output ./my_clips --max-clips 5
  python main.py path/to/video.mp4 --model medium --no-crop --min-duration 20
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig, TranscriberConfig, AnalyzerConfig, ClipperConfig
from transcriber import transcribe, Transcript
from analyzer import ContentAnalyzer
from clipper import VideoClipper


def setup_logging(verbose: bool = True):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def print_banner():
    """Print startup banner."""
    print("""
╔══════════════════════════════════════════════════╗
║           VIDEO AUTO-CLIPPER                     ║
║  Transcript Analysis → Hook Detection → Split    ║
║  Output: 9:16 Vertical Clips for Social Media    ║
╚══════════════════════════════════════════════════╝
    """)


def print_summary(results: list, output_dir: str, elapsed: float):
    """Print final summary."""
    success = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print("\n" + "=" * 55)
    print("  EXTRACTION COMPLETE")
    print("=" * 55)
    print(f"  Clips created:  {len(success)}")
    if failed:
        print(f"  Failed:         {len(failed)}")
    print(f"  Output folder:  {output_dir}")
    print(f"  Time elapsed:   {elapsed:.1f}s")
    print()

    if success:
        print("  Generated clips:")
        for r in success:
            print(
                f"    [{r.clip_number:02d}] {r.start:.1f}s - {r.end:.1f}s "
                f"({r.duration:.0f}s) score={r.score:.3f}"
            )
            print(f"         Reason: {r.reason}")
            print(f"         File: {Path(r.output_path).name}")
            print()

    if failed:
        print("  Failed clips:")
        for r in failed:
            print(f"    [{r.clip_number:02d}] Error: {r.error}")

    print("=" * 55)


def run_pipeline(config: PipelineConfig):
    """Execute the full video analysis + clipping pipeline."""
    logger = logging.getLogger("pipeline")
    start_time = time.time()

    video_path = config.input_video
    output_dir = config.output_dir

    # Validate input
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ─── Stage 1: Transcribe ───
    print("\n[1/3] TRANSCRIBING VIDEO...")
    print(f"  Model: {config.transcriber.model_size}")
    print(f"  This may take a while for long videos...\n")

    transcript = transcribe(
        video_path=video_path,
        model_size=config.transcriber.model_size,
        language=config.transcriber.language,
        device=config.transcriber.device,
    )

    print(f"  Transcription done: {len(transcript.segments)} segments")
    print(f"  Language detected: {transcript.language}")
    print(f"  Video duration: {transcript.duration:.0f}s "
          f"({transcript.duration / 60:.1f} min)\n")

    # Save transcript if requested
    if config.save_analysis:
        transcript_path = str(Path(output_dir) / "transcript.json")
        transcript.save(transcript_path)
        # Also save readable text version
        text_path = str(Path(output_dir) / "transcript.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            for seg in transcript.segments:
                ts = f"[{seg.start:.1f}s - {seg.end:.1f}s]"
                f.write(f"{ts} {seg.text}\n")
        print(f"  Transcript saved: {transcript_path}")

    # ─── Stage 2: Analyze for Hooks ───
    print("\n[2/3] ANALYZING CONTENT FOR HOOKS...")
    analyzer = ContentAnalyzer(config.analyzer)
    candidates = analyzer.analyze(transcript)

    if not candidates:
        print("  No hook-worthy moments found above threshold.")
        print(f"  Try lowering min_hook_score (currently {config.analyzer.min_hook_score})")
        sys.exit(0)

    print(f"  Found {len(candidates)} clip candidates:\n")
    for i, c in enumerate(candidates, 1):
        print(f"  [{i}] {c.start:.1f}s - {c.end:.1f}s "
              f"(score={c.score:.3f}, reason={c.reason})")
        print(f"      \"{c.hook_text[:100]}...\"\n")

    # Save analysis
    if config.save_analysis:
        analysis_path = str(Path(output_dir) / "analysis.json")
        analyzer.save_analysis(candidates, analysis_path)

    # ─── Stage 3: Extract Clips ───
    print("\n[3/3] SPLITTING VIDEO INTO CLIPS...")
    print(f"  Format: {'9:16 vertical' if config.clipper.crop_vertical else 'original aspect ratio'}")
    print(f"  Duration: {config.clipper.min_clip_duration}-{config.clipper.max_clip_duration}s")
    print(f"  Quality: CRF {config.clipper.video_quality}\n")

    clipper = VideoClipper(config.clipper)
    results = clipper.extract_all_clips(
        video_path=video_path,
        candidates=candidates,
        output_dir=output_dir,
        video_duration=transcript.duration,
    )

    # Save results
    if config.save_analysis:
        results_path = str(Path(output_dir) / "results.json")
        clipper.save_results(results, results_path)

    # Print summary
    elapsed = time.time() - start_time
    print_summary(results, output_dir, elapsed)

    return results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Video Auto-Clipper: Analyze content and split into viral short clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py video.mp4
  python main.py video.mp4 --output ./clips --max-clips 5
  python main.py video.mp4 --model medium --language en
  python main.py video.mp4 --no-crop --min-duration 20 --max-duration 90
  python main.py video.mp4 --min-score 0.3 --quality 20
        """
    )

    # Required
    parser.add_argument("video", help="Path to the input video file")

    # Output
    parser.add_argument("-o", "--output", default="./clips_output",
                        help="Output directory for clips (default: ./clips_output)")

    # Transcription
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--language", default=None,
                        help="Language code (e.g., 'en'). Auto-detect if not set")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device for Whisper (default: auto — uses GPU if available)")

    # Analysis
    parser.add_argument("--max-clips", type=int, default=10,
                        help="Maximum number of clips to extract (default: 10)")
    parser.add_argument("--min-score", type=float, default=0.4,
                        help="Minimum hook score threshold 0-1 (default: 0.4)")

    # Clipping
    parser.add_argument("--min-duration", type=int, default=15,
                        help="Minimum clip duration in seconds (default: 15)")
    parser.add_argument("--max-duration", type=int, default=60,
                        help="Maximum clip duration in seconds (default: 60)")
    parser.add_argument("--no-crop", action="store_true",
                        help="Don't crop to 9:16 vertical — keep original aspect ratio")
    parser.add_argument("--quality", type=int, default=23,
                        help="Video quality CRF (lower=better, 18-28, default: 23)")
    parser.add_argument("--no-fade", action="store_true",
                        help="Disable fade in/out transitions")

    # Misc
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save intermediate analysis JSON files")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Minimal output")

    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    print_banner()
    setup_logging(verbose=not args.quiet)

    # Build config from args
    config = PipelineConfig(
        input_video=args.video,
        output_dir=args.output,
        transcriber=TranscriberConfig(
            model_size=args.model,
            language=args.language,
            device=args.device,
        ),
        analyzer=AnalyzerConfig(
            min_hook_score=args.min_score,
            max_clips=args.max_clips,
        ),
        clipper=ClipperConfig(
            min_clip_duration=args.min_duration,
            max_clip_duration=args.max_duration,
            crop_vertical=not args.no_crop,
            video_quality=args.quality,
            fade_duration=0 if args.no_fade else 0.5,
        ),
        save_analysis=not args.no_save,
        verbose=not args.quiet,
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
