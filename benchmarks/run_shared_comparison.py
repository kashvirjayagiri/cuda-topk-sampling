#!/usr/bin/env python3
"""Shared full-sort / hierarchical / radix / PyTorch Top-k comparison runner."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from shared_pipeline import ComparisonLibrary, METHODS, Pipeline, TimedCall, measure_interleaved
from shared_comparison_utils import (
    assert_unchanged, load_dataset, preflight_cases, reference_table, sha256,
    shared_randoms, sliced_reference, timing_stats, validate_output, window_starts)

ROOT = Path(__file__).resolve().parents[1]


def command(*arguments):
    try:
        result = subprocess.run(arguments, cwd=ROOT, text=True, capture_output=True, timeout=15)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def save_metadata(path, metadata):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def validate_pipeline(pipeline, expected, raw, u, stream, operation=None):
    (operation or pipeline.pipeline)()
    stream.synchronize()
    actual = pipeline.snapshot()
    errors = validate_output(expected, actual, u)
    assert_unchanged(pipeline.logits.cpu().numpy(), raw, "Input logits")
    assert_unchanged(pipeline.randoms.cpu().numpy(), u, "Random values")
    return errors


def prepare_operations(pipelines, stream, args):
    operations = {}
    for method, pipeline in pipelines.items():
        for scope, function in (("selection", pipeline.select), ("pipeline", pipeline.pipeline)):
            operations[method, scope] = TimedCall(
                pipeline, function, stream, args.timing, args.calls_per_sample)
        if method == "pytorch":
            operations[method, "raw_topk"] = TimedCall(
                pipeline, pipeline.selector.raw_topk, stream, args.timing, args.calls_per_sample)
    return operations


def run_preflight(library, device, stream, args):
    import torch
    results = []
    for name, raw, k, temperature, u in preflight_cases():
        refs = reference_table(raw, [k], [temperature])
        expected = sliced_reference(refs[k, temperature], 0, len(raw), u)
        logits = torch.from_numpy(raw.copy()).to(device)
        randoms = torch.from_numpy(u.copy()).to(device)
        pipelines = {method: Pipeline(library, method, logits, randoms, k, temperature, stream)
                     for method in METHODS}
        for method, pipeline in pipelines.items():
            validate_pipeline(pipeline, expected, raw, u, stream)
        operations = prepare_operations(pipelines, stream, args)
        # Replaying captured work with NEW data catches stale-address/data bugs.
        for variant, changed in enumerate((raw, raw[:, ::-1].copy())):
            logits.copy_(torch.from_numpy(changed))
            reference = reference_table(changed, [k], [temperature])[k, temperature]
            key = sliced_reference(reference, 0, len(changed), u)
            for method, pipeline in pipelines.items():
                operations[method, "selection"]()
                pipeline.sample()
                stream.synchronize()
                validate_output(key, pipeline.snapshot(), u)
                errors = validate_pipeline(pipeline, key, changed, u, stream,
                                           operations[method, "pipeline"])
                results.append(dict(case=name, variant=variant, method=method,
                                    status="passed", **errors))
        stream.synchronize()
        del operations, pipelines, pipeline, logits, randoms
    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", nargs="+", type=Path,
                        help="Saved logits directories/files; accepts real and synthetic directories together")
    source.add_argument("--smoke-test", action="store_true",
                        help="Generated setup workload, not real-model results")
    p.add_argument("--groups", nargs="+", help="Optional input filename stems, e.g. 01_everyday")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1,8,32])
    p.add_argument("--k", "--k-values", nargs="+", type=int, default=[1,10,50,100])
    p.add_argument("--temperatures", nargs="+", type=float, default=[1.])
    p.add_argument("--windows", default="all", help="all (default) or maximum window count per B")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repetitions", type=int, default=101)
    p.add_argument("--timing", choices=("graph", "eager"), default="graph")
    p.add_argument("--calls-per-sample", type=int, default=1)
    p.add_argument("--seed", type=int, default=451)
    p.add_argument("--random-margin", type=float, default=5e-5,
                   help="Rejection margin from all reference CDF boundaries; 0 gives unconditioned uniforms")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--library", type=Path, default=ROOT/"build/libshared_comparison_python.so")
    p.add_argument("--output", type=Path, help="Fresh results directory, never overwritten")
    args = p.parse_args()
    if any(x < 1 or x > 2147483647 for x in args.batch_sizes):
        p.error("Batch sizes must be positive int32 values")
    if any(x < 1 or x > 256 for x in args.k):
        p.error("Four-method comparison requires k in 1..256 (hierarchical selector limit)")
    for values in (args.batch_sizes, args.k, args.temperatures):
        if len(set(values)) != len(values):
            p.error("Duplicate grid parameters are not allowed")
    with np.errstate(over="ignore", invalid="ignore"):
        args.temperatures = [float(np.float32(t)) for t in args.temperatures]
    if (any(not math.isfinite(t) or t <= 0 for t in args.temperatures)
            or len(set(args.temperatures)) != len(args.temperatures)):
        p.error("Temperatures must be distinct finite positive FP32 values")
    if args.windows != "all":
        try:
            if int(args.windows) < 1: raise ValueError()
        except ValueError:
            p.error("--windows must be all or a positive count")
    if not 0 <= args.warmup <= 100000 or not 1 <= args.repetitions <= 100000:
        p.error("Invalid warmup/repetition count")
    if not 1 <= args.calls_per_sample <= 10000 or args.seed < 0 or args.device < 0:
        p.error("Invalid calls-per-sample, seed, or CUDA device")
    if not 0 <= args.random_margin < .01:
        p.error("Require 0<=random-margin<.01")
    if args.smoke_test and args.groups:
        p.error("--groups applies to saved datasets")
    return args


def discover_files(args):
    files = []
    for path in args.data or []:
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in (".pt", ".pth", ".npy", ".npz")))
        elif path.is_file():
            files.append(path)
        else:
            raise ValueError(f"Missing data path: {path}")
    files = list(dict.fromkeys(p.resolve() for p in files))
    if args.groups:
        unknown = set(args.groups) - {p.stem for p in files}
        if unknown: raise ValueError(f"Unknown groups: {sorted(unknown)}")
        files = [p for p in files if p.stem in args.groups]
    if not args.smoke_test and not files:
        raise ValueError("No saved logits found; pull the committed data first")
    return files


def metadata_for_run(args, library, files):
    import torch
    properties = torch.cuda.get_device_properties(args.device)
    sources = ["src/full_sort.cu", "src/hierarchical_topk.cu", "src/radix_topk.cu", "src/sampler.cu",
               "src/shared_comparison_api.cu", "include/topk.cuh", "include/sampler.h",
               "include/full_sort_sampling.cuh", "include/hierarchical_topk_sampling.cuh",
               "benchmarks/shared_pipeline.py", "benchmarks/torch_topk_adapter.py",
               "benchmarks/shared_comparison_utils.py", "benchmarks/run_shared_comparison.py",
               "benchmarks/summarize_shared_comparison.py", "scripts/build_shared_comparison.sh",
               "CMakeLists.txt"]
    return dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
        methods=list(METHODS), common_sampler="jjk::sample_parallel from src/sampler.cu",
        inputs=[dict(path=str(p), sha256=sha256(p)) for p in files], datasets=[],
        batch_sizes=args.batch_sizes, k=args.k, temperatures=args.temperatures,
        windows=args.windows, warmup=args.warmup, repetitions=args.repetitions,
        method_order="Rotate method/scope order every repetition and input window",
        timing_backend=args.timing, calls_per_sample=args.calls_per_sample,
        statistic="median of individual-call CUDA event times" if args.calls_per_sample == 1
                  else "median of per-sample average per-call CUDA event times",
        timing_scope="CUDA graph replay" if args.timing == "graph" else "eager API submission",
        exclusions=["disk I/O", "model inference", "CPU reference", "validation", "random generation",
                    "explicit allocations", "input/output transfers", "graph capture/instantiation"],
        timing_caveat="Event intervals can include GPU idle gaps while the host submits a replay/call. Graph mode minimizes different Python/native dispatch costs; eager mode includes them and may include PyTorch internal workspace allocator activity.",
        pytorch_selection="GPU torch.topk + kth-boundary tie repair + canonical score/ID ordering + int32 conversion; all included",
        raw_topk="Separate diagnostic only; not substituted for contract-complete selection",
        memory_scope="Explicit selector workspace tensor bytes only; excludes inputs/outputs, graph pools, PyTorch internal temporaries, allocator reservation, and process peak",
        seed=args.seed, random_margin=args.random_margin,
        random_policy="Uniform FP32 draws conditioned away from the union of CDF boundaries" if args.random_margin
                      else "Unconditioned uniform FP32 draws",
        reference="Independent stable CPU sort and NumPy FP64 softmax/CDF, exact original-ID sample",
        atol=1e-5, rtol=1e-5, gpu=properties.name,
        compute_capability=[properties.major, properties.minor], gpu_total_memory=properties.total_memory,
        torch=str(torch.__version__), torch_cuda_build=torch.version.cuda,
        native_versions=library.versions(), python=sys.version, numpy=np.__version__,
        hostname=platform.node(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        git_commit=command("git", "rev-parse", "HEAD"), git_status=command("git", "status", "--porcelain"),
        nvcc_version=command("nvcc", "--version"),
        source_sha256={s: sha256(ROOT/s) for s in sources if (ROOT/s).is_file()},
        library_sha256=sha256(args.library), method_rows_written=0, configurations_completed=0)


def run(args):
    import torch
    files = discover_files(args)
    if not args.library.is_file():
        raise ValueError("Build the shared bridge first: bash scripts/build_shared_comparison.sh")
    if not torch.cuda.is_available():
        raise ValueError("Use an allocated CARC NVIDIA GPU session with CUDA-enabled PyTorch")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.ones(1, device=device).sum().item()  # Check real PyTorch GPU execution.
    library = ComparisonLibrary(args.library)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT/"results"/f"shared-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    metadata = metadata_for_run(args, library, files)
    meta_path = output/"metadata.json"
    save_metadata(meta_path, metadata)
    summary_writer = None
    try:
        with torch.no_grad(), torch.cuda.stream(stream), \
             (output/"summary.csv").open("w", newline="") as summary_file, \
             gzip.open(output/"raw_timings.jsonl.gz", "wt", encoding="utf-8") as raw_file:
            print(f"Results directory: {output}", flush=True)
            print(f"Timing: {args.timing}; same GPU inputs, uniforms, stream and parallel sampler", flush=True)
            print("Checking all four integrations before benchmarking...", flush=True)
            metadata["preflight"] = run_preflight(library, device, stream, args)
            save_metadata(meta_path, metadata)
            print(f"PASS: {len(metadata['preflight'])} preflight method/variant checks", flush=True)
            sources = files or [None]
            for path in sources:
                if path is None:
                    raw = np.random.default_rng(args.seed).normal(
                        size=(max(32, max(args.batch_sizes)), max(1025, max(args.k)))).astype(np.float32)
                    info = dict(name="smoke_synthetic", kind="smoke_synthetic", distribution="normal",
                                path=None, shape=list(raw.shape), source_sha256=__import__("hashlib").sha256(raw.tobytes()).hexdigest())
                    print("SMOKE TEST: generated setup data, not real-model experiment results", flush=True)
                else:
                    raw, info = load_dataset(path)
                for B in args.batch_sizes: window_starts(len(raw), B, args.windows)
                if max(args.k) > raw.shape[1]: raise ValueError(f"{info['name']}: k exceeds V")
                references = reference_table(raw, args.k, args.temperatures)
                all_u = shared_randoms(references, len(raw), args.seed, info["source_sha256"], args.random_margin)
                info["random_values"] = all_u.tolist()
                metadata["datasets"].append(info)
                save_metadata(meta_path, metadata)
                print(f"\nDataset {info['name']}: {tuple(raw.shape)} {info['kind']}", flush=True)
                for B in args.batch_sizes:
                    starts = window_starts(len(raw), B, args.windows)
                    for k in args.k:
                        for temperature in args.temperatures:
                            # Fixed addresses allow reusing captured work on every row window.
                            logits = torch.from_numpy(raw[:B].copy()).to(device)
                            randoms = torch.from_numpy(all_u[:B].copy()).to(device)
                            pipelines = {m: Pipeline(library, m, logits, randoms, k, temperature, stream)
                                         for m in METHODS}
                            first = sliced_reference(references[k, temperature], 0, B, all_u[:B])
                            for pipeline in pipelines.values():
                                validate_pipeline(pipeline, first, raw[:B], all_u[:B], stream)
                            operations = prepare_operations(pipelines, stream, args)
                            for start in starts:
                                metadata["current_case"] = dict(dataset=info["name"], B=B,
                                    V=int(raw.shape[1]), k=k, temperature=temperature, row_start=start)
                                window, u = raw[start:start+B].copy(), all_u[start:start+B].copy()
                                logits.copy_(torch.from_numpy(window))
                                randoms.copy_(torch.from_numpy(u))
                                expected = sliced_reference(references[k, temperature], start, B, u)
                                checked = {}
                                for method, pipeline in pipelines.items():
                                    metadata["current_method"] = method
                                    # Validate the actual captured/eager path and input handoff.
                                    operations[method, "selection"]()
                                    pipeline.sample()
                                    stream.synchronize()
                                    validate_output(expected, pipeline.snapshot(), u)
                                    checked[method] = validate_pipeline(pipeline, expected, window, u,
                                        stream, operations[method, "pipeline"])
                                times = measure_interleaved(operations, stream, args.warmup,
                                    args.repetitions, args.calls_per_sample, metadata["configurations_completed"])
                                case_rows = []
                                for method, pipeline in pipelines.items():
                                    # Do not rerun before checking the last timed outputs.
                                    stream.synchronize()
                                    checked[method] = validate_output(expected, pipeline.snapshot(), u)
                                    assert_unchanged(logits.cpu().numpy(), window, "Input logits")
                                    assert_unchanged(randoms.cpu().numpy(), u, "Random values")
                                    row = dict(dataset=info["name"], dataset_kind=info["kind"],
                                        distribution=info["distribution"], source_sha256=info["source_sha256"],
                                        B=B, V=int(raw.shape[1]), k=k, temperature=temperature,
                                        row_start=start, method=method, correctness="passed",
                                        timing_backend=args.timing, calls_per_sample=args.calls_per_sample,
                                        warmup=args.warmup, repetitions=args.repetitions)
                                    for scope in ("selection", "pipeline"):
                                        row.update({f"{scope}_{s}": value for s, value in
                                            timing_stats(times[method, scope]).items()})
                                    row["raw_topk_median_us"] = (timing_stats(times[method, "raw_topk"])["median_us"]
                                                               if method == "pytorch" else "")
                                    row["pipeline_rows_per_second"] = B*1e6/row["pipeline_median_us"]
                                    row.update(workspace_bytes=pipeline.workspace_bytes, input_bytes=B*raw.shape[1]*4,
                                               random_input_bytes=B*4,
                                               selected_output_bytes=B*k*8, sampler_output_bytes=B*k*8+B*4,
                                               **checked[method])
                                    case_rows.append(row)
                                    record = dict(configuration={name: row[name] for name in
                                        ("dataset", "source_sha256", "B", "V", "k", "temperature", "row_start", "method")},
                                        raw_times_us={scope: values for (m, scope), values in times.items() if m == method},
                                        validation=checked[method], sampled_ids=pipeline.sampled_ids.cpu().tolist())
                                    raw_file.write(json.dumps(record, allow_nan=False)+"\n")
                                if summary_writer is None:
                                    summary_writer = csv.DictWriter(summary_file, fieldnames=list(case_rows[0]))
                                    summary_writer.writeheader()
                                summary_writer.writerows(case_rows)
                                summary_file.flush()
                                raw_file.flush()
                                metadata["configurations_completed"] += 1
                                metadata["method_rows_written"] += len(case_rows)
                                save_metadata(meta_path, metadata)
                                timing_line = " | ".join(f"{r['method']}={r['pipeline_median_us']:.3f} us" for r in case_rows)
                                print(f"  B={B} k={k} T={temperature:g} rows={start}:{start+B} | {timing_line}", flush=True)
                            stream.synchronize()
                            del operations, pipelines, pipeline, logits, randoms
            metadata["status"] = "complete"
    except BaseException as error:
        metadata.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_metadata(meta_path, metadata)
    from summarize_shared_comparison import summarize_run
    summarize_run(output, plots=False)
    print(f"\nPASS: {metadata['method_rows_written']} method/case comparisons; all four pipelines checked")
    print(f"Saved combined results to {output}")
    print(f"Create figures: python3 benchmarks/summarize_shared_comparison.py '{output}' --plots")
    return 0


def main():
    args = parse_args()
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, OverflowError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if args.timing == "graph":
            print("If CUDA graph capture itself failed, use a fresh --output directory with --timing eager and report that timing mode. There is no silent fallback.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
