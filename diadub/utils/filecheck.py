"""
Quick manual artifact verification tool.
Usage:
    python -m diadub.utils.filecheck data/cache/pipeline.checkpoint.json --deep
"""

import argparse
from diadub.storage.checkpoint import CheckpointManager


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Path to checkpoint JSON")
    p.add_argument(
        "--deep", action="store_true", help="Compute SHA-256 for each artifact"
    )
    args = p.parse_args()

    ckpt = CheckpointManager(args.checkpoint)
    ok = ckpt.verify_artifacts(deep=args.deep, log_path="integrity.log")
    print("✅ All good" if ok else "⚠️ Some artifacts failed -- see integrity.log")


if __name__ == "__main__":
    main()
