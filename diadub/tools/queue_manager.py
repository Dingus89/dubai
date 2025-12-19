"""
Simple device-aware task queue for heavy jobs (demucs, TTS inference, vocoder).
- One GPU worker (serial) to avoid OOM and enable device affinity.
- ThreadPoolExecutor for CPU-bound jobs (parallel).
- Basic GPU free-memory probe (nvidia-smi fallback -> torch).
"""

from concurrent.futures import ThreadPoolExecutor, Future
import subprocess
import logging
import time
import os
import torch

log = logging.getLogger("diadub.tools.queue_manager")


class QueueManager:
    def __init__(self, cpu_workers: int = None, gpu_count: int = 1, gpu_mem_threshold_mb: int = 1500):
        self.cpu_workers = cpu_workers or max(1, (os.cpu_count() or 4) // 2)
        self.cpu_pool = ThreadPoolExecutor(max_workers=self.cpu_workers)
        self.gpu_count = gpu_count
        self.gpu_mem_threshold_mb = gpu_mem_threshold_mb
        self.gpu_busy = [False] * self.gpu_count

    def _get_gpu_free_mem(self, gpu_index: int = 0) -> int:
        """Try nvidia-smi first, fallback to torch if available."""
        try:
            out = subprocess.check_output([
                "nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader", "-i", str(
                    gpu_index)
            ])
            return int(out.decode().strip().splitlines()[0])
        except Exception:
            try:
                import torch
                if torch.cuda.is_available():
                    idx = gpu_index
                    # estimate free = total - allocated - reserved
                    total = torch.cuda.get_device_properties(
                        idx).total_memory // (1024*1024)
                    alloc = torch.cuda.memory_allocated(idx) // (1024*1024)
                    reserved = torch.cuda.memory_reserved(idx) // (1024*1024)
                    free = int(total - alloc - reserved)
                    return max(0, free)
            except Exception:
                return 0

    def submit_cpu(self, fn, *args, **kwargs) -> Future:
        return self.cpu_pool.submit(fn, *args, **kwargs)

    def submit_gpu(self, fn, *args, gpu_index: int = 0, wait_for_free: bool = True, **kwargs) -> Future:
        """
        Schedule a GPU task. If GPU appears free (free mem > threshold), dispatch immediately.
        Otherwise either wait (poll) or raise depending on wait_for_free.
        Only one gpu worker is used per gpu_index (simple lock).
        """
        # simple lock loop
        while True:
            free = self._get_gpu_free_mem(gpu_index)
            if free >= self.gpu_mem_threshold_mb and not self.gpu_busy[gpu_index]:
                # claim
                self.gpu_busy[gpu_index] = True
                break
            if not wait_for_free:
                raise RuntimeError("GPU not available")
            log.debug("Waiting for GPU%d free memory >= %d MB (have %d MB)",
                      gpu_index, self.gpu_mem_threshold_mb, free)
            time.sleep(1.0)

        # run the job synchronously in a thread to ensure we can release busy flag on completion
        def wrapper(*a, **k):
            try:
                return fn(*a, **k)
            finally:
                self.gpu_busy[gpu_index] = False

        return self.cpu_pool.submit(wrapper, *args, **kwargs)

    def shutdown(self):
        self.cpu_pool.shutdown(wait=True)
