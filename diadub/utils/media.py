from pathlib import Path
import subprocess
import logging
log = logging.getLogger("diadub.utils.media")


def extract_audio(input_video: str, output_wav: str, sr: int = 48000):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-f", "wav",
        output_wav
    ]
    subprocess.run(cmd, check=True)


def extract_audio_segment(src: str, dst: str, start: float, end: float, sr: int = 16000):
    duration = max(0.0, float(end) - float(start))
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ss", f"{float(start):.3f}", "-t", f"{duration:.3f}",
        "-ac", "1", "-ar", str(sr),
        str(dst)
    ]
    subprocess.run(cmd, check=True)


def get_audio_duration(src: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "format=duration", "-of", "default=nk=1:nw=1", str(src)]
    import subprocess
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode == 0:
        try:
            return float(p.stdout.decode().strip())
        except Exception:
            return 0.0
    return 0.0


def replace_audio(video_path: str, audio_path: str, output_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        str(output_path)
    ]
    subprocess.run(cmd, check=True)
