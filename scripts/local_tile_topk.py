"""ctypes bridge to John's fixed-size local tile Top-k CUDA stage."""
import ctypes
from pathlib import Path

import torch


class LocalTileTopK:
    """Own reusable outputs for one (B, V, k, device); calls use the current stream.

    Repeated calls overwrite outputs. Keep the input alive through stream completion,
    and use a separate instance for concurrent streams.
    """

    def __init__(self, batch_size, vocab_size, k, device, library=None):
        if not (0 < batch_size <= 2**31 - 1 and 0 < vocab_size <= 2**31 - 1
                and 1 <= k <= min(vocab_size, 256)):
            raise ValueError('Require positive int32 B/V and 1 <= k <= min(V, 256)')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('LocalTileTopK requires a CUDA device')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.shape = (batch_size, vocab_size)
        default_library = Path(__file__).resolve().parents[1] / 'build/liblocal_tile_topk_python.so'
        self.library = ctypes.CDLL(str(Path(library or default_library).resolve()))
        self.library.local_tile_count.argtypes = [ctypes.c_int]
        self.library.local_tile_count.restype = ctypes.c_int
        self.library.local_tile_topk_run.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        self.library.local_tile_topk_run.restype = ctypes.c_int
        self.library.local_tile_topk_error_string.argtypes = [ctypes.c_int]
        self.library.local_tile_topk_error_string.restype = ctypes.c_char_p
        tile_count = self.library.local_tile_count(vocab_size)
        if not tile_count:
            raise ValueError('Unsupported vocabulary size')
        self.scores = torch.empty((batch_size, tile_count, k), device=self.device, dtype=torch.float32)
        self.ids = torch.empty((batch_size, tile_count, k), device=self.device, dtype=torch.int32)
        self.k = k

    def __call__(self, logits):
        if (tuple(logits.shape) != self.shape or logits.dtype != torch.float32
                or logits.device != self.device or not logits.is_contiguous()):
            raise ValueError('Expected contiguous CUDA FP32 input matching the configured shape/device')
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            error = self.library.local_tile_topk_run(
                logits.data_ptr(), *self.shape, self.k,
                self.scores.data_ptr(), self.ids.data_ptr(), stream.cuda_stream,
            )
        if error:
            message = self.library.local_tile_topk_error_string(error).decode()
            raise RuntimeError(f'Local tile Top-k CUDA launch failed: {message}')
        return self.scores, self.ids
