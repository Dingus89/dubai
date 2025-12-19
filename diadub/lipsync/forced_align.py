from pathlib import Path
import subprocess
import logging
import json

log = logging.getLogger("diadub.lipsync.forced_align")


def run_mfa_align(audio_wav: str, transcript_srt_or_txt: str, out_dir: str, dict_path: str = None, mfa_binary: str = "mfa"):
    """
    Run Montreal Forced Aligner CLI if available.
    audio_wav: path to the single-segment WAV
    transcript_srt_or_txt: path to transcript text or srt for that segment
    out_dir: target directory for alignment json
    Returns: list of word timings [ {word,start,end} ... ]
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    # Recommended flow: write transcript to a plain .lab file named after wav stem
    stem = Path(audio_wav).stem
    lab = p / f"{stem}.lab"
    with open(lab, "w", encoding="utf-8") as f:
        f.write(open(transcript_srt_or_txt, "r",
                encoding="utf-8").read().strip())

    # mfa align <corpus_dir> <dict> <acoustic_model> <output_dir>
    # But many setups use `mfa_align` high level command. We'll try to run align on single file.
    try:
        cmd = [mfa_binary, "align", str(p), str(dict_path), "english", str(p)]
        subprocess.run(cmd, check=True)
    except Exception as e:
        log.warning("MFA align failed: %s", e)
        return []

    # parse TextGrid (if produced) -- fallback to no timings
    textgrid_path = p / f"{stem}.TextGrid"
    if textgrid_path.exists():
        try:
            return parse_textgrid(textgrid_path)
        except Exception as e:
            log.warning("Failed to parse TextGrid file %s: %s",
                        textgrid_path, e)
            return []

    return []


def parse_textgrid(textgrid_path: Path) -> list:
    with open(textgrid_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    words = []
    in_words_tier = False
    for i, line in enumerate(lines):
        line = line.strip()
        if 'name = "words"' in line:
            in_words_tier = True
        if in_words_tier and "intervals [" in line:
            j = i + 1
            while "]" not in lines[j]:
                xmin_line = lines[j]
                xmax_line = lines[j+1]
                text_line = lines[j+2]
                if "xmin" in xmin_line and "xmax" in xmax_line and "text" in text_line:
                    start_time = float(xmin_line.split("=")[1].strip())
                    end_time = float(xmax_line.split("=")[1].strip())
                    word = text_line.split("=")[1].strip().strip('"')
                    if word:
                        words.append(
                            {"word": word, "start": start_time, "end": end_time})
                j += 3
            break

    return words
