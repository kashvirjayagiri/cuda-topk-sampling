"""Reusable, allocation-free calls to Kashvi's CMake-built full-sort library."""
import ctypes
from pathlib import Path

import torch


class FullSort:
    """Own workspace/output tensors for one fixed (B, V, k, device).

    Repeated calls overwrite the outputs. Input must remain alive until the
    calling CUDA stream completes. Use one instance per concurrent stream.
    Constructor allocation and argument checking are not kernel timings.
    """

    def __init__(self, batch_size, vocab_size, k, device, library=None):
        if not (batch_size > 0 and 1 <= k <= vocab_size <= (1 << 30)):
            raise ValueError('Require B > 0 and 1 <= k <= V <= 2^30')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('FullSort requires a CUDA device')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.shape = (batch_size, vocab_size)
        default_library = Path(__file__).resolve().parents[1] / 'build/libfull_sort_python.so'
        self.library = ctypes.CDLL(str(Path(library or default_library).resolve()))
        self.library.full_sort_workspace_size.argtypes = [ctypes.c_int, ctypes.c_int]
        self.library.full_sort_workspace_size.restype = ctypes.c_size_t
        self.library.full_sort_run.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ]
        self.library.full_sort_run.restype = ctypes.c_int
        self.library.full_sort_error_string.argtypes = [ctypes.c_int]
        self.library.full_sort_error_string.restype = ctypes.c_char_p
        self.capacity = self.library.full_sort_workspace_size(batch_size, vocab_size)
        if not self.capacity:
            raise ValueError('Unsupported workspace dimensions')
        self.work_scores = torch.empty(self.capacity, device=self.device, dtype=torch.float32)
        self.work_ids = torch.empty(self.capacity, device=self.device, dtype=torch.int32)
        self.scores = torch.empty((batch_size, k), device=self.device, dtype=torch.float32)
        self.ids = torch.empty((batch_size, k), device=self.device, dtype=torch.int32)
        self.k = k

    @property
    def workspace_bytes(self):
        return self.work_scores.nbytes + self.work_ids.nbytes

    def __call__(self, logits):
        if (tuple(logits.shape) != self.shape or logits.dtype != torch.float32
                or logits.device != self.device or not logits.is_contiguous()):
            raise ValueError('Expected contiguous CUDA FP32 input matching the configured shape/device')
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            error = self.library.full_sort_run(
                logits.data_ptr(), *self.shape, self.k,
                self.scores.data_ptr(), self.ids.data_ptr(),
                self.work_scores.data_ptr(), self.work_ids.data_ptr(), self.capacity,
                stream.cuda_stream,
            )
        if error:
            message = self.library.full_sort_error_string(error).decode()
            raise RuntimeError(f'Full-sort CUDA launch failed: {message}')
        return self.scores, self.ids

    def benchmark(self, logits, warmup=10, repetitions=50):
        """Return native CUDA-event milliseconds; no allocation in timed regions.

        Event intervals include the entire multi-launch selector, including GPU
        idle gaps if the host cannot issue kernels fast enough. This is not a
        sum of individual kernel durations and does not include sampling.
        """
        if warmup < 0 or repetitions <= 0:
            raise ValueError('Require warmup >= 0 and repetitions > 0')
        # Check input and run once before timing; also initialize CUDA kernels.
        self(logits)
        function = self.library.full_sort_benchmark
        function.argtypes = self.library.full_sort_run.argtypes + [
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
        ]
        function.restype = ctypes.c_int
        samples = (ctypes.c_float * repetitions)()
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            error = function(
                logits.data_ptr(), *self.shape, self.k,
                self.scores.data_ptr(), self.ids.data_ptr(),
                self.work_scores.data_ptr(), self.work_ids.data_ptr(), self.capacity,
                stream.cuda_stream, warmup, repetitions, samples,
            )
        if error:
            message = self.library.full_sort_error_string(error).decode()
            raise RuntimeError(f'Full-sort timing failed: {message}')
        return list(samples)
