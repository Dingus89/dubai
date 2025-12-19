"""
CheckpointManager
Stores and loads pipeline checkpoint JSON files.

Checkpoint layout (example):
{
  "video": "data/samples/test.mp4",
  "stages_completed": {
     "extract_audio": true,
     "asr": true,
     "diarization": false,
     ...
  },
  "artifacts": {
     "audio_path": "output/test.wav",
     "translated_srt": "output/test_translated.srt",
     "synth_files": ["data/cache/test_seg_0000.wav"]
  },
  "meta": {
     "started_at": "2025-10-31T12:00:00Z",
     "last_updated": "..."
  }
}
"""

from pathlib import Path
import json
import tempfile
import time
import hashlib
from typing import Any, Dict, Optional


class CheckpointManager:
    def __init__(self, checkpoint_path: str):
        self.path = Path(checkpoint_path)
        self._data: Dict[str, Any] = {
            "stages_completed": {},
            "artifacts": {},
            "meta": {"created_at": time.time(), "last_updated": time.time()},
        }
        if self.path.exists():
            self.load()

    def load(self) -> Dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        return self._data

    def save(self) -> None:
        # atomic write
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=self.path.name, suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            Path(tmp_path).rename(self.path)
        finally:
            try:
                import os

                os.close(tmp_fd)
            except Exception:
                pass

    def set_stage_done(
        self, stage_name: str, artifacts: Optional[Dict[str, Any]] = None
    ) -> None:
        self._data["stages_completed"][stage_name] = True
        if artifacts:
            # merge artifacts
            for k, v in artifacts.items():
                self._data["artifacts"][k] = v
        self._data["meta"]["last_updated"] = time.time()
        self.save()

    def is_done(self, stage_name: str) -> bool:
        return bool(self._data.get("stages_completed", {}).get(stage_name))

    def get_artifact(self, name: str, default=None):
        return self._data.get("artifacts", {}).get(name, default)

    def set_artifact(self, name: str, value: Any) -> None:
        self._data.setdefault("artifacts", {})[name] = value
        self._data["meta"]["last_updated"] = time.time()
        self.save()

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def remove(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    # --- Progress helpers ---
    def set_progress(self, stage: str, current: int, total: int) -> None:
        """Store numeric progress for a running stage."""
        self._data.setdefault("progress", {})[stage] = {
            "current": current,
            "total": total,
            "ts": time.time(),
        }
        self.save()

    def get_progress(self, stage: str):
        return self._data.get("progress", {}).get(stage)

    # --- Integrity verification ---
    import hashlib

    def _file_ok(self, path: str, deep: bool = False) -> bool:
        p = Path(path)
        if not p.exists() or p.stat().st_size < 1024:  # <1 KB ⇒ bad
            return False
        if deep:
            # compute sha256 for corruption detection
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)
            self._data.setdefault("hashes", {})[str(p)] = h.hexdigest()
            self.save()
        return True

    def verify_artifacts(
        self, deep: bool = False, log_path: Optional[str] = None
    ) -> bool:
        """Check all artifact files. Returns True if all valid."""
        artifacts = self._data.get("artifacts", {})
        failed = []
        for k, v in artifacts.items():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and not self._file_ok(item, deep=deep):
                        failed.append(item)
            elif isinstance(v, str):
                if not self._file_ok(v, deep=deep):
                    failed.append(v)
        if failed:
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{time.ctime()}] Failed artifacts:\n")
                    for x in failed:
                        f.write(f"  {x}\n")
            return False
        return True

    def invalidate_stage(self, stage: str):
        """Mark stage as incomplete (forces rerun)."""
        if stage in self._data.get("stages_completed", {}):
            self._data["stages_completed"][stage] = False
            self.save()
