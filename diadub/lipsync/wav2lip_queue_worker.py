"""
Queue worker for Wav2Lip. Runs as a separate process and reports progress via ProgressManager.
"""

import multiprocessing as mp
import time
import traceback
from pathlib import Path
import logging
import subprocess
from diadub.progress.progress_manager import ProgressManager

log = logging.getLogger("diadub.lipsync.qworker")
log.setLevel(logging.INFO)


def wav2lip_worker(job_queue: mp.Queue, ret_queue: mp.Queue):
    pm = ProgressManager.get()
    while True:
        job = job_queue.get()
        if job is None:
            break
        job_id = job.get("job_id", f"job_{int(time.time())}")
        inf = job["inference_script"]
        face = job["face_video"]
        audio = job["audio"]
        out = job["out"]
        checkpoint = job.get("checkpoint")
        try:
            pm.emit("wav2lip", f"{job_id}: starting", 0.01, {"job_id": job_id})
            cmd = ["python", str(inf), "--face", str(face),
                   "--audio", str(audio), "--outfile", str(out)]
            if checkpoint:
                cmd += ["--checkpoint_path", str(checkpoint)]
            pm.emit("wav2lip", f"{job_id}: running inference", 0.20)
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
            # parse stderr for progress hints (best-effort)
            last_emit = 0.20
            stderr_lines = []
            for line in p.stderr:
                stderr_lines.append(line)
                if "FPS" in line or "frame" in line.lower():
                    last_emit = min(0.7, last_emit + 0.05)
                    pm.emit("wav2lip", f"{job_id}: {line.strip()}", last_emit)
                elif "Writing" in line or "Saved" in line or "Generated" in line:
                    pm.emit("wav2lip", f"{job_id}: finalizing", 0.9)
            code = p.wait()
            if code != 0:
                stderr = "".join(stderr_lines)
                raise RuntimeError(f"Wav2Lip failed (code {code}) {stderr}")
            pm.emit("wav2lip", f"{job_id}: completed", 1.0)
            ret_queue.put({"job_id": job_id, "ok": True, "out": out})
        except Exception as e:
            pm.emit("wav2lip", f"{job_id}: error {e}", None)
            ret_queue.put({"job_id": job_id, "ok": False, "error": str(e)})
            log.error("Worker exception: %s\n%s", e, traceback.format_exc())


class Wav2LipQueueManager:
    def __init__(self, inference_script: str):
        self.job_queue = mp.Queue()
        self.ret_queue = mp.Queue()
        self.process = mp.Process(target=wav2lip_worker, args=(
            self.job_queue, self.ret_queue), daemon=True)
        self.process.start()
        self.inference_script = inference_script

    def submit(self, face_video: str, audio: str, out: str, checkpoint: str = None, job_id: str = None):
        self.job_queue.put({
            "face_video": str(face_video),
            "audio": str(audio),
            "out": str(out),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "job_id": job_id or f"wav2lip_{int(time.time())}",
            "inference_script": str(self.inference_script)
        })

    def get_result(self, timeout: float = None):
        return self.ret_queue.get(timeout=timeout)

    def close(self):
        try:
            self.job_queue.put(None)
            self.process.join(timeout=10)
        except Exception:
            pass
