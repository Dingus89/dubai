"""
aligner.py (final enhanced)
With crossfade stitching + auto-choice between global and per-subsegment DTW alignment
"""

from pathlib import Path
import logging
import math
import tempfile
import shutil

log = logging.getLogger("diadub.alignment")

# --- imports & availability flags ---
_HAS_LIBROSA = _HAS_SOUNDFILE = _HAS_PYRUBBERBAND = _HAS_NUMPY = False
try:
    import numpy as np

    _HAS_NUMPY = True
except Exception as e:
    log.debug("numpy import failed: %s", e)
try:
    import soundfile as sf

    _HAS_SOUNDFILE = True
except Exception as e:
    log.debug("soundfile import failed: %s", e)
try:
    import librosa

    _HAS_LIBROSA = True
except Exception as e:
    log.debug("librosa import failed: %s", e)
try:
    import pyrubberband as pyrb

    _HAS_PYRUBBERBAND = True
except Exception:
    _HAS_PYRUBBERBAND = False
    log.debug("pyrubberband not available")


def _rubberband_cli_available():
    from shutil import which

    return which("rubberband") is not None


def _wav_duration(path: str) -> float:
    try:
        if _HAS_SOUNDFILE:
            info = sf.info(path)
            return info.frames / info.samplerate
        if _HAS_LIBROSA:
            y, sr = librosa.load(path, sr=None, mono=True)
            return len(y) / sr
    except Exception:
        pass
    return 0.0


def compute_mel(
    audio_path: str, sr: int = 22050, n_mels: int = 80, hop_length: int = 512
):
    """Load audio and return mel spectrogram (shape: n_mels x frames)"""
    if not _HAS_LIBROSA:
        raise RuntimeError(
            "librosa not available - install librosa and soundfile for alignment."
        )
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    # convert to log scale (dB) to be more stable
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db, len(y) / sr


def compute_dtw_path(mel_ref, mel_query):
    """Return DTW path aligning mel_ref to mel_query.
    mel_ref and mel_query are 2D arrays (n_mels x frames)."""
    if not _HAS_LIBROSA:
        raise RuntimeError(
            "librosa not available - install librosa and soundfile for alignment."
        )
    # use cosine distance for stability
    D, wp = librosa.sequence.dtw(X=mel_ref, Y=mel_query, metric="cosine")
    # wp is list of (ref_index, query_index) pairs, reverse to forward
    path = np.array(wp)[::-1].T  # shape (2, N)
    ref_idx = path[0].astype(int)
    query_idx = path[1].astype(int)
    return ref_idx, query_idx


def warp_tts_to_target(
    tts_wav_path: str,
    target_duration_s: float,
    out_wav_path: str,
    max_stretch: float = 1.25,
):
    """Time-stretch the TTS waveform to approximately match target_duration_s.

    Uses librosa.effects.time_stretch (phase vocoder). This changes duration but may alter pitch (we do not preserve pitch here).
    For higher quality pitch-preserving stretch, integrate rubberband or pyrubberband.
    """
    if not _HAS_LIBROSA:
        # graceful fallback: copy or pad/truncate
        log.warning(
            "librosa not available; skipping time-stretch. Using naive padding/trimming."
        )
        from shutil import copyfile

        copyfile(tts_wav_path, out_wav_path)
        # try to pad/truncate using soundfile if available
        if _HAS_SOUNDFILE:
            try:
                data, sr = sf.read(out_wav_path)
                cur_dur = len(data) / sr
                if cur_dur < 1e-3:
                    log.warning("TTS audio empty.")
                    return out_wav_path
                if cur_dur < target_duration_s:
                    # pad with silence
                    n_add = int((target_duration_s - cur_dur) * sr)
                    pad = (
                        np.zeros((n_add,) + (data.shape[1:],))
                        if data.ndim > 1
                        else np.zeros(n_add)
                    )
                    new = np.concatenate([data, pad], axis=0)
                    sf.write(out_wav_path, new, sr)
                elif cur_dur > target_duration_s:
                    # truncate
                    new = data[: int(target_duration_s * sr)]
                    sf.write(out_wav_path, new, sr)
            except Exception:
                pass
                return out_wav_path

    # load tts file
    y, sr = librosa.load(tts_wav_path, sr=None, mono=True)
    cur_duration = len(y) / sr
    if cur_duration <= 0:
        log.warning("TTS file empty: %s", tts_wav_path)
        # copy as-is

        shutil.copy(tts_wav_path, out_wav_path)
        return out_wav_path

    # desired stretch factor
    factor = cur_duration / target_duration_s
    # clamp factor
    if factor < 1.0 / max_stretch:
        log.warning(
            "Required compression beyond max_stretch. Clamping factor from %0.3f to %0.3f",
            factor,
            1.0 / max_stretch,
        )
        factor = 1.0 / max_stretch
    if factor > max_stretch:
        log.warning(
            "Required stretching beyond max_stretch. Clamping factor from %0.3f to %0.3f",
            factor,
            max_stretch,
        )
        factor = max_stretch

    if abs(factor - 1.0) < 0.01:
        # close enough
        sf.write(out_wav_path, y, sr)
        return out_wav_path

    # librosa's time_stretch expects frames in the STFT domain --- use phase vocoder wrapper
    try:
        # First compute STFT
        hop_length = 512
        n_fft = 2048
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        D_stretched = librosa.phase_vocoder(
            D, rate=factor, hop_length=hop_length)
        y_stretched = librosa.istft(D_stretched, hop_length=hop_length)
        log.info(f"DEBUG_DURATION: input_duration={cur_duration:.3f}s, target_duration={target_duration_s:.3f}s, factor={factor:.3f}, output_duration={len(y_stretched) / sr:.3f}s")
        sf.write(out_wav_path, y_stretched, sr)
        log.info(
            "Warped %s: %.3fs -> %.3fs (factor=%.3f)",
            tts_wav_path,
            cur_duration,
            len(y_stretched) / sr,
            factor,
        )
        return out_wav_path
    except Exception as e:
        log.exception(
            "Time-stretch failed: %s. Falling back to numpy trim/pad.", e)
        # fallback trim/pad
        if target_duration_s > cur_duration:
            # pad
            n_add = int((target_duration_s - cur_duration) * sr)
            pad = np.zeros(n_add)
            new = np.concatenate([y, pad], axis=0)
            sf.write(out_wav_path, new, sr)
        else:
            new = y[: int(target_duration_s * sr)]
            sf.write(out_wav_path, new, sr)
        return out_wav_path


# -------------------------------------------------------------------------
# Crossfade stitch helper
# -------------------------------------------------------------------------
def _crossfade_concat(chunks, sr: int, fade_ms: float = 20.0):
    """
    Concatenate audio chunks with short linear crossfades to avoid clicks.
    fade_ms: milliseconds of overlap per join.
    """
    if not _HAS_NUMPY or len(chunks) == 0:
        return np.concatenate(chunks, axis=0)

    fade_len = int(sr * fade_ms / 1000.0)
    if fade_len <= 0:
        fade_len = 1

    out = chunks[0].astype(np.float32)
    for next_chunk in chunks[1:]:
        next_chunk = next_chunk.astype(np.float32)
        # overlap fade_len samples (if both long enough)
        f_len = min(fade_len, len(out), len(next_chunk))
        if f_len > 0:
            fade_out = np.linspace(1.0, 0.0, f_len, dtype=np.float32)
            fade_in = 1.0 - fade_out
            out[-f_len:] = out[-f_len:] * fade_out + \
                next_chunk[:f_len] * fade_in
            out = np.concatenate([out, next_chunk[f_len:]], axis=0)
        else:
            out = np.concatenate([out, next_chunk], axis=0)
    return out


# -------------------------------------------------------------------------
# Per-subsegment DTW with crossfade + complexity check
# -------------------------------------------------------------------------
def warp_tts_per_subsegments(
    tts_wav: str,
    ref_wav: str,
    out_wav: str,
    max_stretch: float = 1.5,
    min_chunk_frames=5,
    sr=22050,
    crossfade_ms=20.0,
    auto_choice=True,
):
    """
    Use DTW mel alignment to compute mapping and perform local warps.
    Adds crossfade stitching and auto-choice between global vs. per-segment.
    """
    if not (_HAS_LIBROSA and _HAS_NUMPY and _HAS_SOUNDFILE):
        log.warning("Missing libs for DTW warp; fallback global")
        return warp_tts_to_target(
            tts_wav, _wav_duration(ref_wav), out_wav, max_stretch=max_stretch
        )

    try:
        mel_ref, ref_dur = compute_mel(ref_wav, sr=sr)
        mel_q, q_dur = compute_mel(tts_wav, sr=sr)
        ref_idx, query_idx = compute_dtw_path(mel_ref, mel_q)

        q_to_r = {}
        for r_i, q_i in zip(ref_idx.tolist(), query_idx.tolist()):
            q_to_r.setdefault(q_i, []).append(r_i)
        q_keys = sorted(q_to_r.keys())
        if not q_keys:
            return warp_tts_to_target(
                tts_wav, ref_dur, out_wav, max_stretch=max_stretch
            )

        # build contiguous query segments
        segments = []
        cur_start = q_keys[0]
        cur_end = q_keys[0]
        for k in q_keys[1:]:
            if k == cur_end + 1:
                cur_end = k
            else:
                segments.append((cur_start, cur_end))
                cur_start = k
                cur_end = k
        segments.append((cur_start, cur_end))

        # compute local stretch factors
        hop_length = 512
        factors = []
        for qs, qe in segments:
            r_list = []
            for qf in range(qs, qe + 1):
                r_list.extend(q_to_r.get(qf, []))
            if not r_list:
                continue
            avg_r = sum(r_list) / len(r_list)
            q_time = ((qs + qe) / 2.0) * hop_length / sr
            r_time = avg_r * hop_length / sr
            factors.append((r_time + 1e-6) / (q_time + 1e-6))

        # auto-choice: if variance small, skip DTW warping
        if auto_choice and len(factors) > 1:
            import statistics

            var = statistics.pstdev(factors)
            if var < 0.05:
                log.info("DTW variance %.3f small -- using global stretch", var)
                return warp_tts_to_target(
                    tts_wav, ref_dur, out_wav, max_stretch=max_stretch
                )
            else:
                log.info(
                    "DTW variance %.3f significant -- using per-subsegment warping", var
                )

        # perform chunked warps + crossfade
        y_q, sr_q = sf.read(tts_wav)
        if y_q.ndim > 1:
            y_q = np.mean(y_q, axis=1)

        out_chunks = []
        for qs, qe in segments:
            start_sample = int(qs * hop_length)
            end_sample = int((qe + 1) * hop_length)
            chunk = y_q[start_sample:end_sample]
            if len(chunk) < sr_q * 0.01:
                continue
            # compute local factor using average mapping
            r_list = []
            for qf in range(qs, qe + 1):
                r_list.extend(q_to_r.get(qf, []))
            if not r_list:
                cur_factor = 1.0
            else:
                avg_r = sum(r_list) / len(r_list)
                q_time = ((qs + qe) / 2.0) * hop_length / sr
                r_time = avg_r * hop_length / sr
                cur_factor = (r_time + 1e-6) / (q_time + 1e-6)
                cur_factor = max(1.0 / max_stretch,
                                 min(max_stretch, cur_factor))
            tmp_in = Path(tempfile.mktemp(suffix=".wav"))
            tmp_out = Path(tempfile.mktemp(suffix=".wav"))
            sf.write(str(tmp_in), chunk, sr_q)
            warp_tts_to_target(
                str(tmp_in),
                (len(chunk) / sr_q) * cur_factor,
                str(tmp_out),
                max_stretch=max_stretch,
            )
            warped_chunk, _ = sf.read(str(tmp_out))
            out_chunks.append(warped_chunk)
            tmp_in.unlink(missing_ok=True)
            tmp_out.unlink(missing_ok=True)

        if out_chunks:
            stitched = _crossfade_concat(
                out_chunks, sr_q, fade_ms=crossfade_ms)
            sf.write(out_wav, stitched, sr_q)
            log.info(f"DEBUG_PER_SUBSEGMENT_OUT: Final stitched audio duration: {_wav_duration(str(out_wav)):.3f}s")
            log.info("per-subsegment DTW warp + crossfade done -> %s", out_wav)
            return out_wav
        else:
            return warp_tts_to_target(
                tts_wav, ref_dur, out_wav, max_stretch=max_stretch
            )

    except Exception as e:
        log.exception("warp_tts_per_subsegments failed: %s", e)
        return warp_tts_to_target(
            tts_wav, _wav_duration(ref_wav), out_wav, max_stretch=max_stretch
        )


# -------------------------------------------------------------------------
# align_and_adjust: auto per_subsegment when beneficial
# -------------------------------------------------------------------------
def align_and_adjust(
    tts_wav: str,
    original_segment_wav: str,
    out_wav_path: str,
    max_stretch: float = 1.5,
    per_subsegment: bool = True,
):
    """
    Aligns TTS to original. Automatically decides best method.
    """
    if per_subsegment:
        return warp_tts_per_subsegments(
            tts_wav,
            original_segment_wav,
            out_wav_path,
            max_stretch=max_stretch,
            auto_choice=True,
        )
    else:
        return warp_tts_to_target(
            tts_wav,
            _wav_duration(original_segment_wav),
            out_wav_path,
            max_stretch=max_stretch,
        )
