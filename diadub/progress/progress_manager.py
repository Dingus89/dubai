"""
ProgressManager: central event bus for progress updates.

Features:
 - multiprocessing-safe queue for events
 - subscribe callbacks for GUI or other listeners
 - per-stage timing, ETA estimation based on simple linear extrapolation
 - persistent log file for auditing
"""

import multiprocessing as mp
import time
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Any, Optional

log = logging.getLogger("diadub.progress.manager")
log.setLevel(logging.INFO)


class ProgressManager:
    _instance = None

    def __init__(self, log_path: str = ".diadub_progress.log"):
        self.queue = mp.Queue()
        self.subscribers = []  # subscriber callbacks (callable(event_dict))
        self.log_file = Path(log_path)
        # per-stage timing/progress
        self._stage_state: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get(cls):
        if not cls._instance:
            cls._instance = ProgressManager()
        return cls._instance

    def emit(self, stage: str, message: str, progress: Optional[float] = None, extra: Optional[dict] = None):
        """
        Emit an event. progress is between 0.0 and 1.0 if provided.
        """
        ts = time.time()
        event = {
            "timestamp": ts,
            "stage": stage,
            "message": message,
            "progress": progress,
            "extra": extra or {}
        }

        # maintain stage timing to allow ETA estimation
        state = self._stage_state.setdefault(
            stage, {"start_ts": None, "last_ts": None, "last_progress": None})
        if progress is not None:
            if state["start_ts"] is None:
                state["start_ts"] = ts
            state["last_ts"] = ts
            state["last_progress"] = progress
            # estimate ETA
            eta = self._estimate_eta(stage)
            event["eta_seconds"] = eta

        try:
            self.queue.put(event)
        except Exception:
            log.debug("Failed to put event to queue", exc_info=True)

        # write to log file append-only
        try:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            log.debug("Failed to write progress log", exc_info=True)

        # notify subscribers
        for cb in list(self.subscribers):
            try:
                cb(event)
            except Exception:
                log.debug("Subscriber callback raised", exc_info=True)

    def subscribe(self, callback: Callable[[dict], None]):
        """Subscribe a callback to receive events immediately as they arrive (same-process)."""
        if callback not in self.subscribers:
            self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[dict], None]):
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    def get_queue(self):
        return self.queue

    def _estimate_eta(self, stage: str) -> Optional[float]:
        """
        Very simple linear ETA based on (elapsed / progress) * (1-progress).
        Returns seconds remaining or None if not enough data.
        """
        state = self._stage_state.get(stage)
        if not state:
            return None
        start_ts = state.get("start_ts")
        last_ts = state.get("last_ts")
        last_progress = state.get("last_progress")
        if start_ts is None or last_ts is None or last_progress is None:
            return None
        try:
            elapsed = last_ts - start_ts
            if last_progress <= 0.0:
                return None
            remaining = elapsed * (1.0 - last_progress) / last_progress
            if remaining < 0:
                return 0.0
            return float(remaining)
        except Exception:
            return None
