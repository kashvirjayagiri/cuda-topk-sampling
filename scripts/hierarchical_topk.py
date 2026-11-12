"""ctypes bridge to John's global hierarchical Top-k CUDA selector."""
import ctypes
from pathlib import Path

import torch


class HierarchicalTopK:
    """Own reusable workspace/output tensors for one (B, V, k, device).

    Calls use the current CUDA stream and overwrite outputs. Keep input alive
    through stream completion; use a separate instance for concurrent streams.
    """

    def __init__(self, batch_size, vocab_size, k, device, library=None):
        if not (0 < batch_size <= 2**31 - 1 and 0 < vocab_size <= 2**31 - 1
                and 1 <= k <= min(vocab_size, 256)):
            raise ValueError('Require positive int32 B/V and 1 <= k <= min(V, 256)')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('HierarchicalTopK requires a CUDA device')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.shape = (batch_size, vocab_size)
        default_library = Path(__file__).resolve().parents[1] / 'build/libhierarchical_topk_python.so'
        self.library = ctypes.CDLL(str(Path(library or default_library).resolve()))
        self.library.hierarchical_topk_workspace_size.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.library.hierarchical_topk_workspace_size.restype = ctypes.c_size_t
        self.library.hierarchical_topk_run.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_void_p,
        ]
        self.library.hierarchical_topk_run.restype = ctypes.c_int
        self.library.hierarchical_topk_error_string.argtypes = [ctypes.c_int]
        self.library.hierarchical_topk_error_string.restype = ctypes.c_char_p
        self.capacity = self.library.hierarchical_topk_workspace_size(
            batch_size, vocab_size, k,
        )
        if vocab_size > 256 and not self.capacity:
            raise ValueError('Unsupported workspace dimensions')
        self.work_scores_a = torch.empty(self.capacity, device=self.device, dtype=torch.float32)
        self.work_ids_a = torch.empty(self.capacity, device=self.device, dtype=torch.int32)
        self.work_scores_b = torch.empty(self.capacity, device=self.device, dtype=torch.float32)
        self.work_ids_b = torch.empty(self.capacity, device=self.device, dtype=torch.int32)
        self.scores = torch.empty((batch_size, k), device=self.device, dtype=torch.float32)
        self.ids = torch.empty((batch_size, k), device=self.device, dtype=torch.int32)
        self.k = k

    @property
    def workspace_bytes(self):
        return (self.work_scores_a.nbytes + self.work_ids_a.nbytes
                + self.work_scores_b.nbytes + self.work_ids_b.nbytes)

    def __call__(self, logits):
        if (tuple(logits.shape) != self.shape or logits.dtype != torch.float32
                or logits.device != self.device or not logits.is_contiguous()):
            raise ValueError('Expected contiguous CUDA FP32 input matching the configured shape/device')
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            error = self.library.hierarchical_topk_run(
                logits.data_ptr(), *self.shape, self.k,
                self.scores.data_ptr(), self.ids.data_ptr(),
                self.work_scores_a.data_ptr(), self.work_ids_a.data_ptr(),
                self.work_scores_b.data_ptr(), self.work_ids_b.data_ptr(),
                self.capacity, stream.cuda_stream,
            )
        if error:
            message = self.library.hierarchical_topk_error_string(error).decode()
            raise RuntimeError(f'Hierarchical Top-k CUDA launch failed: {message}')
        return self.scores, self.ids
