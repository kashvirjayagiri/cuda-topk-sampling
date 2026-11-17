"""The same GPU inputs/stream/sampler for all four selection methods."""
import ctypes
from pathlib import Path

METHODS = ("full_sort", "hierarchical", "radix", "pytorch")


class ComparisonLibrary:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.lib = ctypes.CDLL(str(self.path))
        p, i, z, f = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_float
        signatures = {
            "comparison_workspace_elements": ([i, i, i, i], z),
            "comparison_radix_workspace_bytes": ([i, i, i, ctypes.POINTER(z), p], i),
            "comparison_select": ([i, p, i, i, i, p, p, p, p, p, p, z, p], i),
            "comparison_sample": ([p, p, p, i, i, f, p, p, p, p], i),
            "comparison_pipeline": ([i, p, p, i, i, i, f, p, p, p, p, p, p, z, p, p, p, p], i),
            "comparison_error_string": ([i], ctypes.c_char_p),
            "comparison_versions": ([ctypes.POINTER(i), ctypes.POINTER(i)], i),
            "comparison_nvcc_version": ([], ctypes.c_char_p),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, result

    def check(self, status):
        if status:
            message = self.lib.comparison_error_string(status).decode()
            raise RuntimeError(f"Shared CUDA bridge: {message} (code {status})")

    def versions(self):
        runtime, driver = ctypes.c_int(), ctypes.c_int()
        self.check(self.lib.comparison_versions(ctypes.byref(runtime), ctypes.byref(driver)))
        return dict(cuda_runtime=runtime.value, cuda_driver=driver.value,
                    nvcc=self.lib.comparison_nvcc_version().decode())


def pointer(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def tensor_storage_bytes(tensors):
    # Count storage once: views (threshold, arange expansions) share allocations.
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


class Pipeline:
    def __init__(self, library, method, logits, randoms, k, temperature, stream):
        import torch
        if (not logits.is_cuda or logits.dtype != torch.float32 or logits.ndim != 2
                or not logits.is_contiguous() or logits.requires_grad):
            raise ValueError("Require contiguous finite FP32 CUDA logits [B,V]")
        B, V = logits.shape
        if (not 0 < B <= 2147483647 or not 1 <= k <= min(V, 256)
                or V > 2**30 or method not in METHODS):
            raise ValueError("Shared grid requires B>0, V<=2^30 and 1<=k<=min(V,256)")
        if (randoms.device != logits.device or randoms.dtype != torch.float32
                or tuple(randoms.shape) != (B,) or not randoms.is_contiguous()
                or randoms.requires_grad or stream.device != logits.device):
            raise ValueError("Require FP32 randoms [B] and stream on the logits device")
        self.library, self.method, self.stream = library, method, stream
        self.logits, self.randoms = logits, randoms
        self.B, self.V, self.k = B, V, k
        device = logits.device
        self.scores = torch.empty((B, k), dtype=torch.float32, device=device)
        self.ids = torch.empty((B, k), dtype=torch.int32, device=device)
        self.selector = None
        if method == "pytorch":
            from torch_topk_adapter import CanonicalTopK
            self.selector = CanonicalTopK(logits, k)
            self.scores, self.ids = self.selector.scores, self.selector.ids
            tensors = [v for v in vars(self.selector).values() if isinstance(v, torch.Tensor)]
            self.workspace_bytes = tensor_storage_bytes(tensors) - tensor_storage_bytes([logits, self.scores, self.ids])
        self.probabilities = torch.empty_like(self.scores)
        self.cdf = torch.empty_like(self.scores)
        self.sampled_ids = torch.empty((B,), dtype=torch.int32, device=device)
        sp = ctypes.c_void_p(stream.cuda_stream)
        self.sample_arguments = (pointer(self.scores), pointer(self.ids), pointer(randoms),
                                 B, k, temperature, pointer(self.probabilities),
                                 pointer(self.cdf), pointer(self.sampled_ids), sp)
        self.work = []
        if method != "pytorch":
            native = METHODS.index(method)
            if method == "radix":
                size = ctypes.c_size_t()
                library.check(library.lib.comparison_radix_workspace_bytes(
                    B, V, k, ctypes.byref(size), sp))
                capacity = size.value  # Bytes, passed unchanged to RadixWorkspace.
                self.work = [torch.empty(capacity, device=device, dtype=torch.uint8)]
            else:
                capacity = library.lib.comparison_workspace_elements(native, B, V, k)
                if not capacity and (method == "full_sort" or V > 256):
                    raise ValueError("Invalid selector workspace dimensions")
                for _ in range(1 if method == "full_sort" else 2):
                    self.work.extend([torch.empty(capacity, device=device, dtype=torch.float32),
                                      torch.empty(capacity, device=device, dtype=torch.int32)])
            self.workspace_bytes = sum(x.numel()*x.element_size() for x in self.work)
            work_ptrs = [pointer(x) for x in self.work]
            work_ptrs += [ctypes.c_void_p()] * (4 - len(work_ptrs))
            self.select_arguments = (native, pointer(logits), B, V, k,
                                     pointer(self.scores), pointer(self.ids),
                                     *work_ptrs, capacity, sp)
            self.pipeline_arguments = (native, pointer(logits), pointer(randoms), B, V, k,
                                       temperature, pointer(self.scores), pointer(self.ids),
                                       *work_ptrs, capacity, pointer(self.probabilities),
                                       pointer(self.cdf), pointer(self.sampled_ids), sp)

    def select(self):
        if self.selector is not None:
            self.selector.select()
        else:
            self.library.check(self.library.lib.comparison_select(*self.select_arguments))

    def sample(self):
        self.library.check(self.library.lib.comparison_sample(*self.sample_arguments))

    def pipeline(self):
        if self.selector is not None:
            self.select()
            self.sample()
        else:
            # Calls the teammates' actual *_sample integration headers.
            self.library.check(self.library.lib.comparison_pipeline(*self.pipeline_arguments))

    def snapshot(self):
        return {name: tensor.cpu().numpy().copy() for name, tensor in (
            ("scores", self.scores), ("ids", self.ids),
            ("probabilities", self.probabilities), ("cdf", self.cdf),
            ("sampled_ids", self.sampled_ids))}


class TimedCall:
    """Capture once and retain all referenced buffers through their Pipeline.

    Graph replay excludes Python's different kernel-dispatch costs and allocator
    work during capture. Eager mode is available explicitly for API latency.
    """
    def __init__(self, owner, operation, stream, backend, calls):
        import torch
        self.owner, self.operation, self.stream = owner, operation, stream
        self.backend, self.calls, self.graph = backend, calls, None
        for _ in range(3):
            operation()
        stream.synchronize()
        if backend == "graph":
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                for _ in range(calls):
                    operation()

    def __call__(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            for _ in range(self.calls):
                self.operation()


def measure_interleaved(operations, stream, warmup, repetitions, calls, offset=0):
    import torch
    for operation in operations.values():
        for _ in range(warmup):
            operation()
    stream.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(stream)
    end.record(stream)
    end.synchronize()  # Initialize events outside measurements.
    samples = {name: [] for name in operations}
    names = list(operations)
    for repeat in range(repetitions):
        shift = (repeat + offset) % len(names)
        for name in names[shift:] + names[:shift]:
            start.record(stream)
            operations[name]()
            end.record(stream)
            end.synchronize()
            samples[name].append(start.elapsed_time(end)*1000. / calls)
    return samples
