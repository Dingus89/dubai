"""
Simple console watcher that reads events and prints them.
Useful during development or running headless to monitor progress in real-time.
"""

from .progress_manager import ProgressManager
import time
import json
import sys


def watch_progress(poll_rate: float = 0.2):
    pm = ProgressManager.get()
    q = pm.get_queue()
    print("=== diadub Live Progress Monitor ===")
    try:
        while True:
            try:
                ev = q.get(timeout=poll_rate)
            except Exception:
                continue
            ts = time.strftime("%H:%M:%S", time.localtime(ev["timestamp"]))
            stage = ev.get("stage", "")
            msg = ev.get("message", "")
            progress = ev.get("progress")
            eta = ev.get("eta_seconds")
            if progress is not None:
                if eta is not None:
                    print(
                        f"[{ts}] [{stage}] {progress*100:5.1f}% ETA {eta:.1f}s - {msg}")
                else:
                    print(f"[{ts}] [{stage}] {progress*100:5.1f}% - {msg}")
            else:
                print(f"[{ts}] [{stage}] - {msg}")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("Exiting live logger")


if __name__ == "__main__":
    watch_progress()
