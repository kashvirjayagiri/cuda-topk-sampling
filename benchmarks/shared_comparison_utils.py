"""CPU-only inputs, answer keys, and statistics for the shared comparison."""
import hashlib
import math
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def window_starts(rows, batch, windows="all"):
    if batch < 1 or batch > rows:
        raise ValueError(f"B={batch} cannot fit in {rows} rows")
    starts = list(range(0, rows-batch+1, batch))
    if starts[-1] != rows-batch:
        starts.append(rows-batch)  # Cover final rows of a nondivisible dataset.
    if windows != "all":
        count = int(windows)
        if count < 1:
            raise ValueError("--windows must be 'all' or a positive count")
        starts = starts[:count]
    return starts


def finite_logits(raw):
    array = np.asarray(raw)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError("Logits must be real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.ascontiguousarray(array, dtype=np.float32)
    if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError("Require finite FP32 logits [N,V]")
    if array.shape[1] > 2**30:
        raise ValueError("Full-sort supports V<=2^30")
    return array


def load_dataset(path):
    path = Path(path)
    info = dict(name=path.stem, path=str(path.resolve()), source_sha256=sha256(path),
                kind="saved_logits", distribution="unspecified")
    if path.suffix.lower() in (".pt", ".pth"):
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict):
            value = payload.get("logits")
            if "model" in payload:
                info.update(kind="real", distribution="model_outputs", model=str(payload["model"]))
            if "distribution" in payload:
                info.update(kind="synthetic", distribution=str(payload["distribution"]))
        else:
            value = payload
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError(f"{path}: expected a saved FP32 tensor or a dict with FP32 'logits'")
        raw = value.detach().cpu().numpy()
    elif path.suffix.lower() == ".npy":
        raw = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as saved:
            if "logits" not in saved:
                raise ValueError(f"{path}: NPZ requires a logits key")
            raw = saved["logits"].copy()
    else:
        raise ValueError(f"Unsupported input format: {path}")
    raw = finite_logits(raw)
    info["shape"] = list(raw.shape)
    return raw, info


def reference_table(raw, k_values, temperatures):
    """Sort original positions stably; derive FP64 softmax/CDF for each k,T."""
    raw = finite_logits(raw)
    if any(k < 1 or k > raw.shape[1] for k in k_values):
        raise ValueError("k exceeds vocabulary size")
    order = np.argsort(-raw, axis=1, kind="stable")[:, :max(k_values)].astype(np.int32)
    sorted_scores = np.take_along_axis(raw, order, axis=1)
    references = {}
    for k in k_values:
        scores = sorted_scores[:, :k].copy()
        ids = order[:, :k].copy()
        s = scores.astype(np.float64)
        shifted = s-s.max(axis=1, keepdims=True)
        for temperature in temperatures:
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("Invalid temperature")
            with np.errstate(under="ignore", over="ignore"):
                weights = np.exp(shifted/temperature)
            p = weights/weights.sum(axis=1, keepdims=True)
            cdf = np.minimum(np.cumsum(p, axis=1), 1.)
            cdf[:, -1] = 1.
            references[k, temperature] = dict(scores=scores, ids=ids, probabilities=p, cdf=cdf)
    return references


def shared_randoms(references, rows, seed, source_hash, margin=5e-5):
    """One reproducible FP32 uniform per source row, reused for all B/k/T/methods.

    Positive margin uses rejection sampling against the UNION of reference CDF
    boundaries for all requested k/T. This conditions the uniforms for robust
    correctness/timing, not for a distribution-quality experiment. Set margin=0
    for unconditioned uniform draws. The exact accepted values are saved.
    """
    if not 0 <= margin < .01:
        raise ValueError("random margin must be in [0,.01)")
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(source_hash[:8], 16)]))
    u = rng.random(rows, dtype=np.float32)
    if margin == 0:
        return u
    for _ in range(10000):
        invalid = np.zeros(rows, dtype=bool)
        for ref in references.values():
            invalid |= np.min(np.abs(ref["cdf"]-u[:, None]), axis=1) <= margin
        if not invalid.any():
            return u
        u[invalid] = rng.random(int(invalid.sum()), dtype=np.float32)
    raise ValueError("Unable to choose boundary-safe uniforms; reduce --random-margin")


def sliced_reference(reference, start, B, randoms):
    expected = {key: value[start:start+B].copy() for key, value in reference.items()}
    position = np.sum(expected["cdf"] <= randoms[:, None], axis=1)
    expected["sampled_ids"] = expected["ids"][np.arange(B), position]
    return expected


def validate_output(expected, actual, randoms, atol=1e-5, rtol=1e-5):
    for key in ("scores", "ids"):
        if not np.array_equal(actual[key], expected[key]):
            raise ValueError(f"Wrong selected {key}; require score descending, original ID ascending")
    for key in ("probabilities", "cdf"):
        value = actual[key]
        if (value.shape != expected[key].shape or not np.isfinite(value).all()
                or np.any(value < 0) or np.any(value > 1)):
            raise ValueError(f"Invalid {key}: shape/range/finiteness")
        if not np.allclose(value, expected[key], atol=atol, rtol=rtol):
            raise ValueError(f"Wrong {key}: max error {np.max(np.abs(value-expected[key])):.9g}")
    p, cdf = actual["probabilities"], actual["cdf"]
    if not np.allclose(p.sum(axis=1), 1., atol=atol, rtol=rtol):
        raise ValueError("Probabilities do not sum to one")
    if np.any(np.diff(cdf, axis=1) < -1e-7) or not np.all(cdf[:, -1] == 1.):
        raise ValueError("CDF must be nondecreasing and end at one")
    position = np.argmax(cdf > randoms[:, None], axis=1)
    own_ids = actual["ids"][np.arange(len(randoms)), position]
    if not np.array_equal(actual["sampled_ids"], own_ids):
        raise ValueError("Sample does not use first CDF strictly greater than u")
    if not np.array_equal(actual["sampled_ids"], expected["sampled_ids"]):
        gap = np.min(np.abs(expected["cdf"]-randoms[:, None]))
        raise ValueError(f"Sample differs from FP64 key (nearest boundary gap={gap:.9g}); mismatch rejected")
    return dict(max_probability_error=float(np.max(np.abs(p-expected["probabilities"]))),
                max_cdf_error=float(np.max(np.abs(cdf-expected["cdf"]))))


def timing_stats(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError("Nonpositive/nonfinite timing; try --calls-per-sample 10")
    return dict(median_us=float(np.median(a)), p10_us=float(np.percentile(a, 10)),
                p90_us=float(np.percentile(a, 90)))


def assert_unchanged(actual, expected, name):
    if not np.array_equal(np.asarray(actual).view(np.uint32), np.asarray(expected).view(np.uint32)):
        raise ValueError(f"{name} was modified")


def preflight_cases():
    rng = np.random.default_rng(451)
    normal = rng.normal(size=(2, 1025)).astype(np.float32)
    boundary = -np.arange(513, dtype=np.float32)[None, :]
    boundary[0, 255:257] = 100.
    for name, raw, k, temperature, u in (
        ("tiny_padding", np.array([[1,-2,3,-5,2],[-4,0,0,-3,0]], np.float32), 3, 1., [.8,.5]),
        ("odd_partial_tiles", normal, 100, 2., [.3125,.6875]),
        ("cross_tile_tie_k1", boundary, 1, .5, [.3]),
        ("all_equal_partial_tile", np.zeros((8,259), np.float32), 50, 1., [.3125]*8),
        ("exact_cdf_boundary", np.zeros((2,8), np.float32), 4, 1., [.5,.75]),
        ("signed_zero_kV", np.array([[0.,-0.,0.,-0.,0.,-0.]], np.float32), 6, 1., [.3125]),
        ("finite_extremes", np.array([[np.finfo(np.float32).max,-np.finfo(np.float32).max,0.,1.,-1.]], np.float32), 3, .5, [.3]),
        ("maximum_shared_k", rng.integers(-3,4,size=(2,513)).astype(np.float32), 256, 1., [.3125,.6875])):
        yield name, raw, k, temperature, np.asarray(u, np.float32)
