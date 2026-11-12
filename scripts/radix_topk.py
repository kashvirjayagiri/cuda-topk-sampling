"""Reusable ctypes calls to the exact radix Top-k CUDA selector."""
import ctypes
from pathlib import Path

import torch


class RadixTopK:
    """Own one selector workspace and [B,k] outputs.

    Calls use the current CUDA stream and overwrite outputs. Keep input and this
    instance alive through stream completion; use another instance concurrently.
    """

    def __init__(self, batch_size, vocab_size, k, device, library=None):
        if not (0 < batch_size <= 2**31 - 1 and 0 < vocab_size <= 2**31 - 1
                and 1 <= k <= vocab_size and batch_size * vocab_size <= 2**31 - 1):
            raise ValueError('Require B,V > 0, 1 <= k <= V, and B*V <= INT_MAX')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('RadixTopK requires a CUDA device')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.shape = (batch_size, vocab_size)
        self.k = k
        default_library = Path(__file__).resolve().parents[1] / 'build/libradix_topk_python.so'
        self.library = ctypes.CDLL(str(Path(library or default_library).resolve()))
        self.library.radix_topk_workspace_size.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p,
        ]
        self.library.radix_topk_workspace_size.restype = ctypes.c_int
        self.library.radix_topk_run.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_void_p,
        ]
        self.library.radix_topk_run.restype = ctypes.c_int
        self.library.radix_topk_error_string.argtypes = [ctypes.c_int]
        self.library.radix_topk_error_string.restype = ctypes.c_char_p
        size = ctypes.c_size_t()
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            error = self.library.radix_topk_workspace_size(
                batch_size, vocab_size, k, ctypes.byref(size), stream.cuda_stream,
            )
        if error:
            message = self.library.radix_topk_error_string(error).decode()
            raise RuntimeError(f'Radix Top-k workspace query failed: {message}')
        self.workspace = torch.empty(size.value, device=self.device, dtype=torch.uint8)
        self.scores = torch.empty((batch_size, k), device=self.device, dtype=torch.float32)
        self.ids = torch.empty((batch_size, k), device=self.device, dtype=torch.int32)

    @property
    def workspace_bytes(self):
        return self.workspace.nbytes

    def __call__(self, logits):
        if (tuple(logits.shape) != self.shape or logits.dtype != torch.float32
                or logits.device != self.device or not logits.is_contiguous()):
            raise ValueError('Expected contiguous CUDA FP32 input matching configured shape/device')
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            error = self.library.radix_topk_run(
                logits.data_ptr(), *self.shape, self.k,
                self.scores.data_ptr(), self.ids.data_ptr(),
                self.workspace.data_ptr(), self.workspace_bytes, stream.cuda_stream,
            )
        if error:
            message = self.library.radix_topk_error_string(error).decode()
            raise RuntimeError(f'Radix Top-k CUDA call failed: {message}')
        return self.scores, self.ids
