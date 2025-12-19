"""
Map word/phoneme timings to visemes for lip-sync.

Input: word_entries = [ {"word":"hi", "start":0.0,"end":0.22,"phones":["HH","AY"]}, ... ]
Output: {"visemes":[{"viseme":"aa", "start":0.0,"end":0.05}, ...], "phonemes": [...]}
"""

from typing import List, Dict, Any

# Simple CMU phoneme -> viseme coarse mapping (edit as needed)
PHONEME_TO_VISEME = {
    # vowels
    "AA": "AA", "AE": "AA", "AH": "AH", "AO": "AA", "AW": "AW", "AY": "AY",
    "EH": "EH", "ER": "ER", "EY": "EY", "IH": "IH", "IY": "IY", "OW": "OW", "OY": "OY", "UH": "UH", "UW": "UW",
    # consonants map to closed/open shapes
    "P": "MBP", "B": "MBP", "M": "MBP",
    "F": "FV", "V": "FV",
    "TH": "TH", "DH": "TH",
    "T": "TD", "D": "TD", "S": "SZ", "Z": "SZ", "SH": "SH", "ZH": "SH",
    "CH": "CH", "JH": "CH",
    "K": "KG", "G": "KG", "NG": "KG",
    "L": "L", "R": "R",
    "W": "W", "Y": "Y", "HH": "HH",
    # fallback
}


def phonemes_to_visemes(phonemes: List[str]) -> List[str]:
    return [PHONEME_TO_VISEME.get(p.upper(), "rest") for p in phonemes]


def generate_viseme_timing(word_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    visemes = []
    phonemes_out = []
    for w in word_entries:
        start = w.get("start", 0.0)
        end = w.get("end", start + w.get("duration", 0.05))
        phones = w.get("phones") or w.get("phonemes") or []
        phonemes_out.append(
            {"word": w.get("word"), "phones": phones, "start": start, "end": end})
        vm = phonemes_to_visemes(phones)
        # spread viseme timings evenly across the word duration
        if vm:
            seg = (end - start) / max(1, len(vm))
            t = start
            for v in vm:
                visemes.append({"viseme": v, "start": round(
                    t, 6), "end": round(t + seg, 6)})
                t += seg
        else:
            visemes.append({"viseme": "rest", "start": start, "end": end})
    return {"visemes": visemes, "phonemes": phonemes_out}


def apply_visemes_to_video(video_path: str, viseme_data: List[Dict], output_path: str):
    """
    Placeholder function for applying viseme data to a video.
    This function should take the viseme data and overlay corresponding
    visuals (e.g., mouth shapes) onto the video at specified timings.
    """
    import logging
    log = logging.getLogger("diadub.lipsync.viseme_mapper")
    log.warning(
        "Lip-sync application to video is not yet implemented in this function.")
    log.info(f"Viseme data generated for {video_path}: {viseme_data}")
    # In a real implementation, this would involve ffmpeg commands
    # or a dedicated video processing library to alter the video frame by frame.
