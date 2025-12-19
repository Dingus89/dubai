"""
Lightweight face tracker to generate per-character timelines.
Prefer insightface for embeddings; fallback on face_recognition / Haar cascade.
"""

from pathlib import Path
import logging
import numpy as np
import cv2
from collections import defaultdict

log = logging.getLogger("diadub.lipsync.character_tracker")
log.setLevel(logging.INFO)

try:
    from insightface.app import FaceAnalysis
    _HAS_INSIGHT = True
except Exception:
    _HAS_INSIGHT = False

try:
    import face_recognition
    _HAS_FR = True
except Exception:
    _HAS_FR = False

_cascade = None
if not _HAS_INSIGHT and not _HAS_FR:
    try:
        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:
        _cascade = None


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class CharacterTracker:
    def __init__(self, device: str = "cpu", min_face_size: int = 40):
        self.device = device
        self.min_face_size = min_face_size
        self._backend = None
        self._init_backend()

    def _init_backend(self):
        if _HAS_INSIGHT:
            try:
                self.app = FaceAnalysis(name="buffalo_l")
                ctx_id = 0 if self.device in ("cuda", "gpu") else -1
                self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                self._backend = "insight"
                log.info("InsightFace backend ready")
                return
            except Exception:
                log.exception("InsightFace init failed")
        if _HAS_FR:
            self._backend = "face_recognition"
            log.info("Using face_recognition backend")
            return
        if _cascade:
            self._backend = "haar"
            log.info("Using Haar cascade backend")
            return
        self._backend = "none"
        log.warning("No face backend available; tracker disabled")

    def analyze_video(self, video_path: str, sample_rate: int = 2):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        tracks = []
        next_id = 0
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue

            detections = []
            if self._backend == "insight":
                try:
                    faces = self.app.get(frame)
                    for f in faces:
                        x1, y1, x2, y2 = map(int, f.bbox)
                        w, h = max(1, x2 - x1), max(1, y2 - y1)
                        if w < self.min_face_size:
                            continue
                        emb = None
                        try:
                            emb = f.normed_embedding.astype(np.float32)
                        except Exception:
                            emb = None
                        detections.append(
                            {"bbox": (x1, y1, w, h), "emb": emb, "score": float(f.det_score or 0.0)})
                except Exception:
                    log.debug("insight detect error", exc_info=True)
            elif self._backend == "face_recognition":
                try:
                    rgb = frame[:, :, ::-1]
                    locs = face_recognition.face_locations(rgb)
                    encs = face_recognition.face_encodings(rgb, locs)
                    for (top, right, bottom, left), emb in zip(locs, encs):
                        w, h = right-left, bottom-top
                        if w < self.min_face_size:
                            continue
                        detections.append(
                            {"bbox": (left, top, w, h), "emb": emb.astype(np.float32), "score": 1.0})
                except Exception:
                    log.debug("face_recognition error", exc_info=True)
            elif self._backend == "haar":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(
                    self.min_face_size, self.min_face_size))
                for (x, y, w, h) in rects:
                    detections.append(
                        {"bbox": (int(x), int(y), int(w), int(h)), "emb": None, "score": 0.5})
            else:
                frame_idx += 1
                continue

            # match detections to tracks
            for det in detections:
                assigned = False
                for t in tracks:
                    last_frame, last_bbox, _ = t["frames"][-1]
                    score = _iou(det["bbox"], last_bbox)
                    if score > 0.3:
                        t["frames"].append(
                            (frame_idx, det["bbox"], det.get("score", 0.0)))
                        if det.get("emb") is not None:
                            if t.get("emb") is None:
                                t["emb"] = det["emb"]
                            else:
                                t["emb"] = 0.95 * t["emb"] + 0.05 * det["emb"]
                        assigned = True
                        break
                if not assigned:
                    tracks.append({"track_id": next_id, "frames": [
                                  (frame_idx, det["bbox"], det.get("score", 0.0))], "emb": det.get("emb")})
                    next_id += 1

            frame_idx += 1

        cap.release()
        timeline = []
        for t in tracks:
            if not t["frames"]:
                continue
            first = t["frames"][0][0]
            last = t["frames"][-1][0]
            timeline.append({
                "char_id": f"char_{t['track_id']:02d}",
                "frames": t["frames"],
                "first_frame": int(first),
                "last_frame": int(last),
                "embedding": t.get("emb").tolist() if t.get("emb") is not None else None
            })
        return {"fps": fps, "timeline": timeline}
