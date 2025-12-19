import sys
import os
import json
import threading
import subprocess
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any

# PyQt5 imports (replace with PySide2/6 if you prefer)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFileDialog, QProgressBar, QTextEdit, QGroupBox, QComboBox,
    QCheckBox, QListWidget, QListWidgetItem, QMessageBox, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer

# ---------------------------
# Internal app imports (safe)
# ---------------------------
try:
    from diadub.pipeline.pipeline import run_full_pipeline
except Exception:
    run_full_pipeline = None

try:
    from diadub.models.wav2lip.wav2lip_wrapper import run_wav2lip
except Exception:
    run_wav2lip = None

try:
    from diadub.queue.queue_manager import QueueManager
except Exception:
    QueueManager = None

try:
    from diadub.progress.progress_manager import ProgressManager
    pm = ProgressManager.get()
except Exception:
    ProgressManager = None
    pm = None

# progress callback helpers; GUI will use local if these aren't present
try:
    from diadub.progress.progress_callbacks import gui_progress_callback, gui_eta_callback
except Exception:
    gui_progress_callback = None
    gui_eta_callback = None

try:
    from diadub.utils.gpu import detect_gpu_info
except Exception:
    def detect_gpu_info():
        # fallback: try torch
        try:
            import torch
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                prop = torch.cuda.get_device_properties(idx)
                return {"name": prop.name, "vram_gb": round(prop.total_memory / (1024**3), 2)}
        except Exception:
            pass
        return None


# ---------------------------
# Helper wrappers for safety
# ---------------------------

def safe_submit_job(task_name: str, func, kwargs: dict, progress_cb=None, eta_cb=None, on_complete=None, on_error=None):
    """
    Submit a job via QueueManager if available; otherwise run in a background thread.
    Returns a job_id (string) or None.
    """
    if QueueManager is not None:
        try:
            return QueueManager.submit(
                task_name=task_name,
                func=func,
                kwargs=kwargs,
                progress_cb=progress_cb,
                eta_cb=eta_cb,
                on_complete=on_complete,
                on_error=on_error
            )
        except Exception as e:
            # fall back to thread
            print("QueueManager.submit failed; falling back to thread:", e)

    # thread fallback
    job_id = f"thread_{int(time.time()*1000)}"

    def worker():
        try:
            result = func(**kwargs)
            if on_complete:
                try:
                    on_complete(result)
                except Exception:
                    pass
        except Exception as ex:
            if on_error:
                try:
                    on_error(ex)
                except Exception:
                    pass
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return job_id


def safe_cancel_job(job_id: str) -> bool:
    if QueueManager is not None:
        try:
            if hasattr(QueueManager, "cancel"):
                return QueueManager.cancel(job_id)
        except Exception:
            pass
    # if thread fallback, cancellation not supported
    return False


def format_seconds(s: Optional[float]) -> str:
    if s is None:
        return "--:--"
    s = max(0, int(round(s)))
    m = s // 60
    sec = s % 60
    return f"{m:02d}:{sec:02d}"


# ---------------------------
# GUI Main Window
# ---------------------------

class JobItem:
    """Simple container for job metadata (for GUI list)."""

    def __init__(self, job_id: str, job_type: str, kwargs: Dict[str, Any]):
        self.job_id = job_id
        self.job_type = job_type
        self.kwargs = kwargs
        self.status = "queued"
        self.progress = 0.0
        self.eta = None
        self.started_at = None
        self.finished_at = None
        self.result = None

    def to_display(self) -> str:
        p = int(self.progress*100) if self.progress is not None else 0
        eta = format_seconds(self.eta)
        return f"{self.job_id} | {self.job_type} | {self.status} | {p}% | ETA {eta}"


class DiadubGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIA-DUB -- Final GUI")
        self.setMinimumSize(900, 700)

        self.job_registry: Dict[str, JobItem] = {}
        self._build_ui()
        self.gpu_info = detect_gpu_info()
        self._update_gpu_label()
        self._setup_timer()

    # --------------------------
    # UI construction
    # --------------------------
    def _build_ui(self):
        root = QVBoxLayout()

        # --- top: file selectors & options ---
        top_box = QGroupBox("Inputs & Options")
        top_layout = QVBoxLayout()

        # Video selection
        row = QHBoxLayout()
        lbl = QLabel("Video:")
        self.video_path_input = QLineEdit()
        btn = QPushButton("Browse")
        btn.clicked.connect(self._choose_video)
        row.addWidget(lbl)
        row.addWidget(self.video_path_input)
        row.addWidget(btn)
        top_layout.addLayout(row)

        # SRT (optional)
        row = QHBoxLayout()
        lbl = QLabel("SRT (optional):")
        self.srt_path_input = QLineEdit()
        btn = QPushButton("Browse")
        btn.clicked.connect(self._choose_srt)
        row.addWidget(lbl)
        row.addWidget(self.srt_path_input)
        row.addWidget(btn)
        top_layout.addLayout(row)

        # Output selection
        row = QHBoxLayout()
        lbl = QLabel("Output file:")
        self.output_path_input = QLineEdit()
        btn = QPushButton("Browse")
        btn.clicked.connect(self._choose_output)
        row.addWidget(lbl)
        row.addWidget(self.output_path_input)
        row.addWidget(btn)
        top_layout.addLayout(row)

        # Model selector, options
        row = QHBoxLayout()
        lbl = QLabel("TTS Model:")
        self.model_select = QComboBox()
        self.model_select.addItems(["VibeVoice-1.5B", "Orpheus-3B-quant"])
        row.addWidget(lbl)
        row.addWidget(self.model_select)

        self.checkbox_wav2lip = QCheckBox("Run Wav2Lip (optional)")
        self.checkbox_save_clips = QCheckBox("Save speaker clips for cloning")
        self.checkbox_gpu_safe = QCheckBox("GPU-safe (auto VRAM fallback)")
        self.checkbox_wav2lip.setChecked(True)
        self.checkbox_gpu_safe.setChecked(True)

        row.addWidget(self.checkbox_wav2lip)
        row.addWidget(self.checkbox_save_clips)
        row.addWidget(self.checkbox_gpu_safe)

        top_layout.addLayout(row)

        # Wav2Lip checkpoint path
        row = QHBoxLayout()
        lbl = QLabel("Wav2Lip checkpoint:")
        self.wav2lip_ckpt = QLineEdit(
            str(Path("diadub/models/wav2lip/checkpoints/wav2lip_gan.pth")))
        btn = QPushButton("Browse")
        btn.clicked.connect(self._choose_wav2lip_ckpt)
        row.addWidget(lbl)
        row.addWidget(self.wav2lip_ckpt)
        row.addWidget(btn)
        top_layout.addLayout(row)

        # GPU Info Label
        self.gpu_label = QLabel("GPU: Detecting...")
        self.gpu_label.setStyleSheet("font-style: italic; color: #888;")
        top_layout.addWidget(self.gpu_label, alignment=Qt.AlignRight)

        top_box.setLayout(top_layout)
        root.addWidget(top_box)

        # --- Middle: Control buttons ---
        controls = QGroupBox("Actions")
        c_layout = QHBoxLayout()
        self.btn_run_pipeline = QPushButton("Run Full Pipeline")
        self.btn_run_pipeline.clicked.connect(self.on_run_pipeline_clicked)
        self.btn_stop_pipeline = QPushButton("Cancel Selected Job")
        self.btn_stop_pipeline.clicked.connect(self.on_cancel_selected_job)
        self.btn_run_wav2lip = QPushButton("Run Wav2Lip (only)")
        self.btn_run_wav2lip.clicked.connect(self.on_run_wav2lip_clicked)
        self.btn_preview = QPushButton("Preview Video")
        self.btn_preview.clicked.connect(self.on_preview_clicked)

        c_layout.addWidget(self.btn_run_pipeline)
        c_layout.addWidget(self.btn_run_wav2lip)
        c_layout.addWidget(self.btn_stop_pipeline)
        c_layout.addWidget(self.btn_preview)
        controls.setLayout(c_layout)
        root.addWidget(controls)

        # --- Lower: Dashboard & progress ---
        dashboard = QGroupBox("Job Dashboard")
        d_layout = QHBoxLayout()

        # Job list
        self.job_list = QListWidget()
        self.job_list.itemSelectionChanged.connect(
            self.on_job_selection_changed)
        d_layout.addWidget(self.job_list, 3)

        # details & per-stage bars
        right_panel = QVBoxLayout()

        self.lbl_job_details = QLabel("Select a job to see details")
        right_panel.addWidget(self.lbl_job_details)

        self.stage_progress_box = QVBoxLayout()
        # dynamic per-stage bars will be stored in a dict
        self.stage_bars: Dict[str, QProgressBar] = {}
        right_panel.addLayout(self.stage_progress_box)

        # ETA label
        self.lbl_eta = QLabel("ETA: --:--")
        right_panel.addWidget(self.lbl_eta)

        d_layout.addLayout(right_panel, 2)
        dashboard.setLayout(d_layout)
        root.addWidget(dashboard)

        # --- Log panel ---
        logs = QGroupBox("Log")
        logs_layout = QVBoxLayout()
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        logs_layout.addWidget(self.log_panel)

        # log controls
        lr = QHBoxLayout()
        self.btn_clear_log = QPushButton("Clear Log")
        self.btn_clear_log.clicked.connect(self.clear_log)
        self.btn_save_log = QPushButton("Save Log")
        self.btn_save_log.clicked.connect(self.save_log)
        lr.addWidget(self.btn_clear_log)
        lr.addWidget(self.btn_save_log)
        logs_layout.addLayout(lr)

        logs.setLayout(logs_layout)
        root.addWidget(logs)

        # Set layout
        self.setLayout(root)

    def _update_gpu_label(self):
        if self.gpu_info:
            name = self.gpu_info.get("name", "Unknown GPU")
            vram = self.gpu_info.get("vram_gb", "?")
            self.gpu_label.setText(f"GPU: {name} ({vram} GB VRAM)")
            self.gpu_label.setStyleSheet("color: #4a4;")
        else:
            self.gpu_label.setText("GPU: Not detected (or PyTorch not installed with CUDA)")
            self.gpu_label.setStyleSheet("color: #a44;")

    # --------------------------
    # File chooser helpers
    # --------------------------
    def _choose_video(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select video", "", "Video files (*.mp4 *.mov *.mkv)")
        if p:
            self.video_path_input.setText(p)

    def _choose_srt(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select SRT", "", "Subtitle files (*.srt)")
        if p:
            self.srt_path_input.setText(p)

    def _choose_output(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "Select output", "", "Video files (*.mp4)")
        if p:
            self.output_path_input.setText(p)

    def _choose_wav2lip_ckpt(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Wav2Lip checkpoint", "", "PTH files (*.pth)")
        if p:
            self.wav2lip_ckpt.setText(p)

    # --------------------------
    # Controls: actions
    # --------------------------
    def on_run_pipeline_clicked(self):
        video = self.video_path_input.text().strip()
        output = self.output_path_input.text().strip()
        srt = self.srt_path_input.text().strip() or None
        if not video or not output:
            QMessageBox.warning(
                self, "Missing", "Please select input video and output path.")
            return

        kwargs = {
            "video_path": video,
            "output_path": output,
            "srt_path": srt,
            "use_wav2lip": bool(self.checkbox_wav2lip.isChecked()),
            "save_speaker_clips": bool(self.checkbox_save_clips.isChecked()),
            "gpu_safe": bool(self.checkbox_gpu_safe.isChecked()),
            "tts_model": self.model_select.currentText(),
            "wav2lip_checkpoint": self.wav2lip_ckpt.text().strip(),
        }

        job_id = safe_submit_job(
            task_name="full_pipeline",
            func=run_full_pipeline if run_full_pipeline else self._fallback_run_pipeline,
            kwargs=kwargs,
            progress_cb=lambda stage, pct: self._on_progress_callback(
                job_id=job_id if False else None, stage=stage, pct=pct),
            eta_cb=lambda stage, eta: self._on_eta_callback(
                job_id=job_id if False else None, stage=stage, eta=eta),
            on_complete=lambda r: self._on_job_complete(job_id, r),
            on_error=lambda e: self._on_job_error(job_id, e)
        )

        # create JobItem and add to GUI registry
        # note: safe_submit_job returns job_id for QueueManager, else thread fallback returns a generated id
        if job_id is None:
            job_id = f"job_{int(time.time()*1000)}"
        ji = JobItem(job_id=job_id, job_type="full_pipeline", kwargs=kwargs)
        ji.status = "submitted"
        self.job_registry[job_id] = ji
        self._add_job_list_item(ji)
        self.log(f"Submitted pipeline job {job_id}")

    def on_run_wav2lip_clicked(self):
        video = self.video_path_input.text().strip()
        audio = self.audio_path_input.text().strip()
        output = self.output_path_input.text().strip()
        ckpt = self.wav2lip_ckpt.text().strip()
        if not video or not audio or not output:
            QMessageBox.warning(
                self, "Missing", "Please select video, audio, and output for Wav2Lip.")
            return

        kwargs = {
            "video_path": video,
            "audio_path": audio,
            "output_path": output,
            "checkpoint_path": ckpt or str(Path("diadub/models/wav2lip/checkpoints/wav2lip_gan.pth")),
            "device": "cuda" if self.checkbox_gpu_safe.isChecked() else "cpu"
        }

        job_id = safe_submit_job(
            task_name="wav2lip",
            func=run_wav2lip if run_wav2lip else self._fallback_run_wav2lip,
            kwargs=kwargs,
            progress_cb=lambda stage, pct: self._on_progress_callback(
                job_id=None, stage=stage, pct=pct),
            eta_cb=lambda stage, eta: self._on_eta_callback(
                job_id=None, stage=stage, eta=eta),
            on_complete=lambda r: self._on_job_complete(job_id, r),
            on_error=lambda e: self._on_job_error(job_id, e)
        )

        if job_id is None:
            job_id = f"wav2lip_{int(time.time()*1000)}"
        ji = JobItem(job_id=job_id, job_type="wav2lip", kwargs=kwargs)
        ji.status = "submitted"
        self.job_registry[job_id] = ji
        self._add_job_list_item(ji)
        self.log(f"Submitted Wav2Lip job {job_id}")

    def on_cancel_selected_job(self):
        item = self.job_list.currentItem()
        if not item:
            QMessageBox.information(
                self, "Select job", "Please select a job to cancel.")
            return
        job_id = item.data(Qt.UserRole)
        ok = safe_cancel_job(job_id)
        if ok:
            self.log(f"Cancel requested for job {job_id}")
            ji = self.job_registry.get(job_id)
            if ji:
                ji.status = "cancel_requested"
                self._refresh_job_item(job_id)
        else:
            QMessageBox.information(
                self, "Cancel", "Cancel not supported for this job or QueueManager is not available.")

    def on_preview_clicked(self):
        video = self.video_path_input.text().strip()
        if not video:
            QMessageBox.information(self, "Preview", "Select a video first.")
            return
        # open with system default player (safe)
        try:
            if sys.platform == "win32":
                os.startfile(video)
            elif sys.platform == "darwin":
                subprocess.run(["open", video])
            else:
                subprocess.run(["xdg-open", video])
        except Exception as e:
            QMessageBox.warning(self, "Preview failed", str(e))

    # --------------------------
    # Job list helpers
    # --------------------------
    def _add_job_list_item(self, ji: JobItem):
        item = QListWidgetItem(ji.to_display())
        item.setData(Qt.UserRole, ji.job_id)
        self.job_list.addItem(item)

    def _refresh_job_item(self, job_id: str):
        ji = self.job_registry.get(job_id)
        if not ji:
            return
        # find item
        for idx in range(self.job_list.count()):
            itm = self.job_list.item(idx)
            if itm.data(Qt.UserRole) == job_id:
                itm.setText(ji.to_display())
                break
        # update details area if currently selected
        cur = self.job_list.currentItem()
        if cur and cur.data(Qt.UserRole) == job_id:
            self._show_job_details(ji)

    def on_job_selection_changed(self):
        item = self.job_list.currentItem()
        if not item:
            self.lbl_job_details.setText("Select a job to see details")
            return
        job_id = item.data(Qt.UserRole)
        ji = self.job_registry.get(job_id)
        if ji:
            self._show_job_details(ji)

    def _show_job_details(self, ji: JobItem):
        text = json.dumps({
            "job_id": ji.job_id,
            "job_type": ji.job_type,
            "status": ji.status,
            "progress": ji.progress,
            "eta": ji.eta,
            "kwargs": ji.kwargs,
            "started_at": ji.started_at,
            "finished_at": ji.finished_at,
            "result": ji.result
        }, indent=2)
        self.lbl_job_details.setText(text)
        # build/update per-stage bars if present in ji.result (or use generic)
        self._clear_stage_bars()
        # If we have stage info in result.extra, create a bar for each stage
        stages = []
        if isinstance(ji.result, dict):
            stages = ji.result.get("stages", []) or []
        if not stages:
            # create a generic single bar for job progress
            bar = QProgressBar()
            bar.setValue(int(ji.progress*100))
            self.stage_bars["overall"] = bar
            self.stage_progress_box.addWidget(QLabel("Overall"))
            self.stage_progress_box.addWidget(bar)
        else:
            for s in stages:
                name = s.get("name", "stage")
                pct = int((s.get("progress") or 0.0)*100)
                bar = QProgressBar()
                bar.setValue(pct)
                self.stage_bars[name] = bar
                self.stage_progress_box.addWidget(QLabel(name))
                self.stage_progress_box.addWidget(bar)

        self.lbl_eta.setText(f"ETA: {format_seconds(ji.eta)}")

    def _clear_stage_bars(self):
        # remove widgets in the layout
        while self.stage_progress_box.count():
            w = self.stage_progress_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.stage_bars = {}

    # --------------------------
    # Progress callbacks (called by QueueManager or local thread)
    # --------------------------
    def _on_progress_callback(self, job_id: Optional[str], stage: str, pct: Optional[float]):
        # This will be called from background threads/processes; use Qt's signals or invoke via timer
        # We will simply schedule a UI update via QTimer.singleShot
        def _update():
            # If job_id not provided, heuristics: update selected job
            ji = None
            if job_id and job_id in self.job_registry:
                ji = self.job_registry[job_id]
            else:
                # pick selected job
                sel = self.job_list.currentItem()
                if sel:
                    ji = self.job_registry.get(sel.data(Qt.UserRole))
                else:
                    # fallback: any running job
                    for j in self.job_registry.values():
                        if j.status in ("submitted", "running", "started"):
                            ji = j
                            break
            if not ji:
                return
            if pct is None:
                ji.progress = ji.progress or 0.0
            else:
                ji.progress = float(
                    pct if pct <= 1.0 else pct/100.0) if pct is not None else ji.progress
            ji.status = "running"
            if ji.started_at is None:
                ji.started_at = time.time()
            # update displayed bars if this stage exists
            # create per-stage bar if absent
            if stage not in self.stage_bars:
                bar = QProgressBar()
                self.stage_bars[stage] = bar
                self.stage_progress_box.addWidget(QLabel(stage))
                self.stage_progress_box.addWidget(bar)
            val = int((ji.progress or 0.0)*100)
            self.stage_bars[stage].setValue(val)
            self._refresh_job_item(ji.job_id)
            # log
            self.log(f"[{ji.job_id}] {stage} -> {val}%")
        QTimer.singleShot(0, _update)

    def _on_eta_callback(self, job_id: Optional[str], stage: str, eta: Optional[float]):
        def _update():
            ji = None
            if job_id and job_id in self.job_registry:
                ji = self.job_registry[job_id]
            else:
                sel = self.job_list.currentItem()
                if sel:
                    ji = self.job_registry.get(sel.data(Qt.UserRole))
            if not ji:
                return
            ji.eta = eta
            self.lbl_eta.setText(f"ETA: {format_seconds(eta)}")
        QTimer.singleShot(0, _update)

    # --------------------------
    # Job completion & error handlers
    # --------------------------
    def _on_job_complete(self, job_id: Optional[str], result: Any):
        # find ji by job_id (if None, try to find "running" job)
        if job_id is None:
            # try to find some running job
            ji = next((j for j in self.job_registry.values()
                      if j.status in ("submitted", "running")), None)
        else:
            ji = self.job_registry.get(job_id)
        if ji:
            ji.status = "finished"
            ji.finished_at = time.time()
            ji.result = result
            ji.progress = 1.0
            self._refresh_job_item(ji.job_id)
            self.log(f"Job {ji.job_id} finished: {result}")
            # set UI progress to 100
            for b in self.stage_bars.values():
                b.setValue(100)
            self.lbl_eta.setText("ETA: Done")

    def _on_job_error(self, job_id: Optional[str], exc: Exception):
        # similar lookup as complete
        if job_id is None:
            ji = next((j for j in self.job_registry.values()
                      if j.status in ("submitted", "running")), None)
        else:
            ji = self.job_registry.get(job_id)
        if ji:
            ji.status = "error"
            ji.finished_at = time.time()
            ji.result = str(exc)
            self._refresh_job_item(ji.job_id)
            self.log(f"Job {ji.job_id} ERROR: {exc}")
            QMessageBox.warning(self, "Job Error",
                                f"Job {ji.job_id} error: {exc}")

    # --------------------------
    # Logging helpers
    # --------------------------
    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.log_panel.append(f"[{ts}] {msg}")

    def clear_log(self):
        self.log_panel.clear()

    def save_log(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "diadub_log.txt", "Text files (*.txt)")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.log_panel.toPlainText())
            QMessageBox.information(self, "Saved", f"Log saved to {p}")

    # --------------------------
    # Timer & housekeeping
    # --------------------------
    def _setup_timer(self):
        # simple timer to poll QueueManager job list and update statuses
        self._timer = QTimer(self)
        self._timer.setInterval(1000)  # 1s
        self._timer.timeout.connect(self._poll_job_statuses)
        self._timer.start()

    def _poll_job_statuses(self):
        # Query QueueManager (if available) for statuses
        if QueueManager is not None and hasattr(QueueManager, "list_jobs"):
            try:
                jobs = QueueManager.list_jobs()
                # jobs expected as list of dicts: {"job_id":..,"status":..,"progress":..,"eta":..}
                for job in jobs:
                    jid = job.get("job_id")
                    ji = self.job_registry.get(jid)
                    if not ji:
                        ji = JobItem(job_id=jid, job_type=job.get(
                            "task_name", "unknown"), kwargs={})
                        self.job_registry[jid] = ji
                        self._add_job_list_item(ji)
                    ji.status = job.get("status", ji.status)
                    prog = job.get("progress", None)
                    if prog is not None:
                        ji.progress = float(
                            prog if prog <= 1.0 else prog/100.0)
                    ji.eta = job.get("eta_seconds", ji.eta)
                    self._refresh_job_item(jid)
            except Exception:
                # don't spam errors
                pass

    # --------------------------
    # Fallback runners (if modules missing)
    # --------------------------
    def _fallback_run_pipeline(self, video_path: str, output_path: str, **kwargs):
        # Minimal fallback: just extract audio and save a marker file
        outdir = Path(output_path).parent
        outdir.mkdir(parents=True, exist_ok=True)
        marker = outdir / (Path(video_path).stem + "_pipeline_placeholder.txt")
        marker.write_text(
            "Fallback pipeline executed. Please install pipeline module.", encoding="utf-8")
        return {"status": "fallback_done", "marker": str(marker)}

    def _fallback_run_wav2lip(self, video_path: str, audio_path: str, output_path: str, **kwargs):
        # If run_wav2lip missing, attempt to call system-level wav2lip inference if available
        if shutil.which("wav2lip-inference"):
            cmd = ["wav2lip-inference", "--face", video_path,
                   "--audio", audio_path, "--outfile", output_path]
            subprocess.run(cmd)
            return {"status": "invoked_system_wav2lip", "outfile": output_path}
        # else create marker
        outdir = Path(output_path).parent
        outdir.mkdir(parents=True, exist_ok=True)
        marker = outdir / (Path(video_path).stem + "_wav2lip_placeholder.txt")
        marker.write_text(
            "Fallback wav2lip executed. Please install wav2lip module.", encoding="utf-8")
        return {"status": "fallback_wav2lip", "marker": str(marker)}

# --------------------------
# Run GUI
# --------------------------


def run_gui():
    app = QApplication(sys.argv)
    win = DiadubGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()
