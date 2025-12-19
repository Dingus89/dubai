import json
import logging
from pathlib import Path

log = logging.getLogger("diadub.script.parser")


def load_script(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_for_tts(script_json: str):
    s = load_script(script_json)
    out = []
    for ln in s.get("lines", []):
        text = ln.get("translated") or ln.get("text", "")
        start, end = float(ln.get("start", 0)), float(ln.get("end", 0.5))
        if end <= start:
            end = start + 0.5
        tts = ln.get("tts_hint", {})
        out.append(
            {
                "index": ln.get("index"),
                "text": text,
                "start": start,
                "end": end,
                "duration": ln.get("analysis", {}).get("duration", end - start),
                "prosody": {
                    "rate": tts.get("rate", 1.0),
                    "pitch": tts.get("pitch_semitones", 0.0),
                    "gain": tts.get("gain_db", 0.0),
                },
                "voice": tts.get("voice"),
                "emotion": ln.get("emotion"),
            }
        )
    return out
