"""
High-level enhancer that:
 - generates character timeline via CharacterTracker
 - maps script speakers to characters
 - uses Wav2LipQueueManager to process per-segment mouth crops
 - composites enhanced crops back into a working copy of the video
Uses ProgressManager to emit stage progress and ETA.
"""

import argparse
import json
import tempfile
import shutil
import os
from pathlib import Path
import logging
from diadub.progress.progress_manager import ProgressManager
from .character_tracker import CharacterTracker
from .speaker_character_linker import link_speakers_to_characters
from .wav2lip_runner import crop_face_video, composite_crop_back
from .wav2lip_queue_worker import Wav2LipQueueManager

log = logging.getLogger("diadub.lipsync.enhance")
log.setLevel(logging.INFO)
pm = ProgressManager.get()


def enhance_main(script_path: str, input_video: str, tts_audio: str, out_video: str, wav2lip_inference: str = None, sample_rate: int = 2, wav2lip_checkpoint: str = None):
    tmpdir = Path(tempfile.mkdtemp(prefix="diadub_lipsync_"))
    try:
        pm.emit("lipsync", "Analyzing characters in video", 0.01)
        ct = CharacterTracker(device="cuda" if os.getenv(
            "CUDA_VISIBLE_DEVICES") else "cpu")
        char_timeline = ct.analyze_video(
            str(input_video), sample_rate=sample_rate)
        pm.emit("lipsync", "Character analysis complete", 0.10, {
                "num_characters": len(char_timeline.get("timeline", []))})

        pm.emit("lipsync", "Linking script speakers to characters", 0.12)
        mapping, details = link_speakers_to_characters(
            str(script_path), char_timeline)
        pm.emit("lipsync", "Speaker->Character mapping complete",
                0.14, {"mapping": mapping})

        # prepare wav2lip queue manager
        inference = wav2lip_inference or os.getenv(
            "WAV2LIP_INFERENCE") or os.getenv("WAV2LIP_PATH")
        if not inference:
            raise RuntimeError(
                "Wav2Lip inference script not provided (set wav2lip_inference or WAV2LIP_INFERENCE/WAV2LIP_PATH)")
        qm = Wav2LipQueueManager(inference_script=inference)

        script = json.loads(Path(script_path).read_text(encoding="utf-8"))
        segments = script.get("lines", [])
        total = len(segments)
        working_video = tmpdir / "work_video.mp4"
        shutil.copy2(input_video, working_video)

        for idx, seg in enumerate(segments):
            pm.emit(
                "lipsync", f"Processing segment {idx+1}/{total}", idx/total)
            spk = seg.get("speaker")
            if spk not in mapping:
                continue
            char_id = mapping[spk]
            entry = next(
                (c for c in char_timeline["timeline"] if c["char_id"] == char_id), None)
            if not entry or not entry.get("frames"):
                continue
            fps = char_timeline.get("fps", 25.0)
            target_frame = int(round(seg.get("start", 0.0) * fps))
            # find frame nearest to start
            nearest = min(entry["frames"],
                          key=lambda t: abs(t[0] - target_frame))
            _, bbox, _score = nearest
            crop_vid = tmpdir / f"crop_{idx}.mp4"
            crop_face_video(str(working_video), tuple(
                bbox), str(crop_vid), pad=8)
            # extract segment audio
            seg_audio = tmpdir / f"seg_{idx}.wav"
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start + 0.001))
            dur = max(0.001, end - start)
            cmd = ["ffmpeg", "-y", "-i", str(tts_audio), "-ss",
                   f"{start:.3f}", "-t", f"{dur:.3f}", "-ar", "16000", "-ac", "1", str(seg_audio)]
            subprocess = __import__("subprocess")
            p = subprocess.run(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
            if p.returncode != 0:
                pm.emit(
                    "lipsync", f"Failed to extract audio for segment {idx}", None)
                continue
            # submit job
            out_crop = tmpdir / f"enh_crop_{idx}.mp4"
            job_id = f"seg_{idx}"
            pm.emit("wav2lip", f"Submitting job {job_id}", idx/total)
            qm.submit(face_video=str(crop_vid), audio=str(seg_audio), out=str(
                out_crop), checkpoint=wav2lip_checkpoint, job_id=job_id)
            try:
                res = qm.get_result(timeout=300)
            except Exception as e:
                pm.emit("wav2lip", f"Timeout/failure for {job_id}: {e}", None)
                continue
            if not res.get("ok"):
                pm.emit(
                    "wav2lip", f"Wav2Lip reported error for {job_id}: {res.get('error')}", None)
                continue
            pm.emit("wav2lip", f"Wav2Lip finished {job_id}", (idx+0.5)/total)
            # composite
            new_work = tmpdir / f"work_{idx}.mp4"
            try:
                composite_crop_back(str(working_video), str(
                    out_crop), tuple(bbox), str(new_work))
                shutil.move(str(new_work), str(working_video))
                pm.emit(
                    "lipsync", f"Segment {idx} composited", (idx+0.85)/total)
            except Exception as e:
                pm.emit(
                    "lipsync", f"Composite failed for segment {idx}: {e}", None)
                continue

        # final copy
        shutil.copy2(working_video, out_video)
        pm.emit("lipsync", "Enhancement complete", 1.0)
        qm.close()
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wav2lip-inference", required=False)
    parser.add_argument("--sample-rate", type=int, default=2)
    parser.add_argument("--wav2lip-checkpoint", default=None)
    args = parser.parse_args()
    enhance_main(args.script, args.video, args.audio, args.out, wav2lip_inference=args.wav2lip_inference,
                 sample_rate=args.sample_rate, wav2lip_checkpoint=args.wav2lip_checkpoint)
