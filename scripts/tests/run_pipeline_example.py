"""
    Small example runner for pipeline. Use for local testing on a short WAV.
    """
import argparse
from pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input WAV")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--base", required=True, help="Base name")
    parser.add_argument("--srt", default=None, help="Optional srt")
    args = parser.parse_args()
    run_pipeline(args.input, args.out, args.base, srt_path=args.srt)
