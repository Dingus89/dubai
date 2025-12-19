from pathlib import Path
import json
import logging
from collections import defaultdict

log = logging.getLogger("diadub.lipsync.speaker_character_linker")


def link_speakers_to_characters(script_path: str, character_timeline: dict, min_overlap_seconds: float = 0.2):
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(script_path)
    script = json.loads(script_path.read_text(encoding="utf-8"))
    lines = script.get("lines", [])
    fps = character_timeline.get("fps", 25.0)
    timeline = character_timeline.get("timeline", [])

    char_windows = {}
    for c in timeline:
        start_s = c["first_frame"] / fps
        end_s = c["last_frame"] / fps
        char_windows[c["char_id"]] = (start_s, end_s)

    overlap_scores = defaultdict(lambda: defaultdict(float))
    for ln in lines:
        spk = ln.get("speaker")
        if not spk:
            continue
        seg_start = float(ln.get("start", 0.0))
        seg_end = float(ln.get("end", seg_start + 0.001))
        for cid, (cstart, cend) in char_windows.items():
            overlap = max(0.0, min(seg_end, cend) - max(seg_start, cstart))
            if overlap >= min_overlap_seconds:
                overlap_scores[spk][cid] += overlap

    mapping = {}
    details = {}
    for spk, scores in overlap_scores.items():
        if not scores:
            details[spk] = {}
            continue
        best_char = max(scores.items(), key=lambda kv: kv[1])[0]
        mapping[spk] = best_char
        details[spk] = dict(scores)

    log.info("Speaker->Char mapping: %s", mapping)
    return mapping, details
