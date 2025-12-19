"""
Small utility to assign and persist a mapping between
speaker names and voice ids. Saves to voices/voice_map.json
"""

import json
from pathlib import Path

VOICE_MAP = Path("voices/voice_map.json")


def load_map():
    if VOICE_MAP.exists():
        return json.loads(VOICE_MAP.read_text(encoding="utf-8"))
    return {}


def save_map(m):
    VOICE_MAP.parent.mkdir(parents=True, exist_ok=True)
    VOICE_MAP.write_text(json.dumps(m, indent=2), encoding="utf-8")


def assign_persistent_voice(speaker_name: str, suggested: str = None) -> str:
    m = load_map()
    if speaker_name in m:
        return m[speaker_name]
    # use suggested, else create a unique id
    vid = suggested if suggested else f"voice_{abs(hash(speaker_name)) % 10000}"
    m[speaker_name] = vid
    save_map(m)
    return vid


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: assign_persistent_voice.py <speaker_name> [voice_id]")
        sys.exit(1)
    name = sys.argv[1]
    suggested = sys.argv[2] if len(sys.argv) > 2 else None
    print(assign_persistent_voice(name, suggested))
