"""
Helpers to crop face region, run Wav2Lip (via inference script), and composite back.
This version is safe and works with the queue manager (job submission).
"""

from pathlib import Path
import os
import subprocess
import logging
import cv2

log = logging.getLogger("diadub.lipsync.wav2lip_runner")
log.setLevel(logging.INFO)


def _find_wav2lip_inference():
    # Look for env var or common repo layout
    env = os.getenv("WAV2LIP_PATH")
    if env:
        candidate = Path(env) / "inference.py"
        if candidate.exists():
            return str(candidate)
    local = Path.cwd() / "Wav2Lip" / "inference.py"
    if local.exists():
        return str(local)
    # last resort: rely on PATH entry/wrapper
    from shutil import which
    alt = which("wav2lip-inference") or which("wav2lip")
    return alt


def crop_face_video(original_video: str, bbox: tuple, out_crop: str, pad: int = 8):
    cap = cv2.VideoCapture(str(original_video))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for cropping")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x, y, bw, bh = bbox
    x0 = max(0, int(x - pad))
    y0 = max(0, int(y - pad))
    x1 = min(w, int(x + bw + pad))
    y1 = min(h, int(y + bh + pad))
    out_w, out_h = x1 - x0, y1 - y0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_crop), fourcc, fps, (out_w, out_h))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y0:y1, x0:x1]
        if crop.shape[0] != out_h or crop.shape[1] != out_w:
            crop = cv2.resize(crop, (out_w, out_h))
        writer.write(crop)
    writer.release()
    cap.release()
    return out_crop


def composite_crop_back(original_video: str, enhanced_crop_video: str, bbox: tuple, out_video: str):
    x, y, w, h = bbox
    # overlay with ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-i", original_video,
        "-i", enhanced_crop_video,
        "-filter_complex", f"[0:v][1:v] overlay={x}:{y}:shortest=1",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        out_video
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(
            f"Composite failed: {p.stderr.decode(errors='ignore')}")
    return out_video
