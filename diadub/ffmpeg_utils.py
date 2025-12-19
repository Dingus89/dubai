import subprocess
import logging
from pathlib import Path

log = logging.getLogger("diadub.ffmpeg")


def extract_audio(video_path: str, output_path: str = None) -> str:
    """Extracts audio from video to a wav file."""
    video_path = Path(video_path)
    output_path = output_path or video_path.with_suffix(".wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "48000",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    log.info(f"Extracted audio: {output_path}")
    return str(output_path)


def replace_audio(video_path: str, new_audio_path: str, output_path: str = None) -> str:
    """Replaces audio track in a video file."""
    video_path = Path(video_path)
    output_path = output_path or video_path.with_name(
        f"{video_path.stem}_dub.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(new_audio_path),
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    log.info(f"Replaced audio: {output_path}")
    return str(output_path)
