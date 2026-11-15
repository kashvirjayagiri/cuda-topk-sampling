"""Reusable current-stream bridge for hierarchical Top-k plus parallel sampling."""
import ctypes
import math
from pathlib import Path

import torch


class HierarchicalTopKSampling:
    """Own reusable selection, sampler scratch, and output tensors for (B,V,k)."""

    def __init__(self, batch_size, vocab_size, k, device, library=None):
        if not (0 < batch_size <= 2**31 - 1 and 0 < vocab_size <= 2**31 - 1
                and 1 <= k <= min(vocab_size, 256)):
            raise ValueError('Require positive int32 B/V and 1 <= k <= min(V,256)')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('HierarchicalTopKSampling requires a CUDA device')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.shape = (batch_size, vocab_size)
        self.k = k
        default_library = Path(__file__).resolve().parents[1] / 'build/libhierarchical_topk_sampling_python.so'
        self.library = ctypes.CDLL(str(Path(library or default_library).resolve()))
        self.library.hierarchical_topk_sampling_workspace_size.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.library.hierarchical_topk_sampling_workspace_size.restype = ctypes.c_size_t
        self.library.hierarchical_topk_sampling_run.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        self.library.hierarchical_topk_sampling_run.restype = ctypes.c_int
        self.library.hierarchical_topk_sampling_error_string.argtypes = [ctypes.c_int]
        self.library.hierarchical_topk_sampling_error_string.restype = ctypes.c_char_p
        self.capacity = self.library.hierarchical_topk_sampling_workspace_size(
            batch_size, vocab_size, k)
        if vocab_size > 256 and not self.capacity:
            raise ValueError('Unsupported workspace dimensions')
        self.work_scores_a = torch.empty(self.capacity, device=self.device, dtype=torch.float32)
        self.work_ids_a = torch.empty(self.capacity, device=self.device, dtype=torch.int32)
        self.work_scores_b = torch.empty(self.capacity, device=self.device, dtype=torch.float32)
        self.work_ids_b = torch.empty(self.capacity, device=self.device, dtype=torch.int32)
        self.selected_scores = torch.empty((batch_size, k), device=self.device, dtype=torch.float32)
        self.selected_ids = torch.empty((batch_size, k), device=self.device, dtype=torch.int32)
        self.probabilities = torch.empty((batch_size, k), device=self.device, dtype=torch.float32)
        self.cdf = torch.empty((batch_size, k), device=self.device, dtype=torch.float32)
        self.sampled_ids = torch.empty(batch_size, device=self.device, dtype=torch.int32)

    def __call__(self, logits, randoms, temperature=1.0):
        if (tuple(logits.shape) != self.shape or logits.dtype != torch.float32
                or logits.device != self.device or not logits.is_contiguous()):
            raise ValueError('Expected contiguous CUDA FP32 logits matching configured shape/device')
        if (tuple(randoms.shape) != (self.shape[0],) or randoms.dtype != torch.float32
                or randoms.device != self.device or not randoms.is_contiguous()):
            raise ValueError('Expected contiguous CUDA FP32 randoms [B] on the same device')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be finite and positive')
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            error = self.library.hierarchical_topk_sampling_run(
                logits.data_ptr(), randoms.data_ptr(), *self.shape, self.k,
                temperature, self.selected_scores.data_ptr(), self.selected_ids.data_ptr(),
                self.work_scores_a.data_ptr(), self.work_ids_a.data_ptr(),
                self.work_scores_b.data_ptr(), self.work_ids_b.data_ptr(), self.capacity,
                self.probabilities.data_ptr(), self.cdf.data_ptr(),
                self.sampled_ids.data_ptr(), stream.cuda_stream)
        if error:
            message = self.library.hierarchical_topk_sampling_error_string(error).decode()
            raise RuntimeError(f'Hierarchical selection plus sampling launch failed: {message}')
        return self.selected_scores, self.selected_ids, self.sampled_ids
