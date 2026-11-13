"""ctypes bridge for direct raw-logit Top-k rejection sampling."""
import ctypes
import math
from pathlib import Path

import torch


class RejectionSampler:
    """Own reusable [B] sampled-ID and attempt-count outputs.

    Calls use the current CUDA stream and overwrite outputs. Keep the input and
    this instance alive until that stream completes; use separate instances for
    concurrent streams.
    """

    def __init__(self, batch_size, vocab_size, k, device, library=None):
        if not (0 < batch_size <= 2**31 - 1 and 0 < vocab_size <= 2**31 - 1
                and 1 <= k <= vocab_size):
            raise ValueError('Require positive int32 B/V and 1 <= k <= V')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('RejectionSampler requires a CUDA device')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.shape = (batch_size, vocab_size)
        self.k = k
        default_library = Path(__file__).resolve().parents[1] / 'build/librejection_sampler_python.so'
        self.library = ctypes.CDLL(str(Path(library or default_library).resolve()))
        self.library.rejection_sample_topk_run.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_ulonglong, ctypes.c_ulonglong,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        self.library.rejection_sample_topk_run.restype = ctypes.c_int
        self.library.rejection_sample_topk_error_string.argtypes = [ctypes.c_int]
        self.library.rejection_sample_topk_error_string.restype = ctypes.c_char_p
        self.ids = torch.empty(batch_size, device=self.device, dtype=torch.int32)
        self.attempts = torch.empty(batch_size, device=self.device, dtype=torch.int32)

    def __call__(self, logits, temperature=1.0, seed=451, draw_index=0):
        if (tuple(logits.shape) != self.shape or logits.dtype != torch.float32
                or logits.device != self.device or not logits.is_contiguous()):
            raise ValueError('Expected contiguous CUDA FP32 input matching configured shape/device')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be finite and positive')
        if not (0 <= seed < 2**64 and 0 <= draw_index < 2**64):
            raise ValueError('seed and draw_index must be unsigned 64-bit integers')
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            error = self.library.rejection_sample_topk_run(
                logits.data_ptr(), *self.shape, self.k, temperature,
                seed, draw_index, self.ids.data_ptr(), self.attempts.data_ptr(),
                stream.cuda_stream,
            )
        if error:
            message = self.library.rejection_sample_topk_error_string(error).decode()
            raise RuntimeError(f'Rejection sampler CUDA launch failed: {message}')
        return self.ids, self.attempts
