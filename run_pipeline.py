# !/usr/bin/env python3
import argparse
import logging
from pathlib import Path

from diadub.utils.logging_config import setup_logging
from diadub.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="DiaDub pipeline runner")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--srt", default=None, help="Optional input SRT file")
    parser.add_argument(
        "--model-config", default="models.json", help="Model registry JSON"
    )
    parser.add_argument("--out-dir", default="output", help="Output directory")
    parser.add_argument(
        "--device", default="cuda", help="Device for models (cuda or cpu)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint if present",
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Path to checkpoint JSON file (optional)"
    )
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("diadub")
    log.info("Starting pipeline for %s", args.video)

    pipeline = Pipeline(
        model_config=args.model_config,
        device=args.device,
        temp_dir="data/cache",
        checkpoint_path=args.checkpoint,
        resume=args.resume,
    )
    result = pipeline.run(
        video_path=args.video, srt_path=args.srt, out_dir=args.out_dir
    )

    log.info("Pipeline result: %s", result)


if __name__ == "__main__":
    main()
