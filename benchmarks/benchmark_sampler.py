#!/usr/bin/env python3
"""Check then time your serial and parallel CUDA samplers. Run from repo root."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from sampler_benchmark_utils import prepare_cases,encode_cases,parse_output,summarize,reference_torch,reference_numpy


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda:source.read(1048576),b""):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--logits",type=Path,help="Actual raw [N,V] logits: .npy, .npz, .pt/.pth")
    source.add_argument("--input",type=Path,help="Preselected NPZ: scores [N,K], token_ids [N,K], random_values [N]")
    source.add_argument("--smoke-test",action="store_true",help="Generated setup data; NOT real-model experiment results")
    parser.add_argument("--executable",type=Path,default=Path("build/sampler_benchmark"))
    parser.add_argument("--batch-sizes",type=int,nargs="+")
    parser.add_argument("--k-values",type=int,nargs="+")
    parser.add_argument("--temperatures",type=float,nargs="+",default=[1.])
    parser.add_argument("--start-row",type=int,default=0)
    parser.add_argument("--windows",type=int,default=1)
    parser.add_argument("--seed",type=int,default=451)
    parser.add_argument("--warmup",type=int,default=20)
    parser.add_argument("--repeats",type=int,default=101)
    parser.add_argument("--launches-per-sample",type=int,default=1)
    parser.add_argument("--reference",choices=("torch","numpy"),default="torch")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if not 0<=args.warmup<=100000 or not 1<=args.repeats<=100000 or not 1<=args.launches_per_sample<=100000:
        parser.error("Invalid warmup/repetition/launch count")
    if not args.executable.is_file():parser.error("Compile build/sampler_benchmark first (see README)")
    torch_version=None
    torch_threads=None
    if args.reference=="torch":
        try:import torch
        except ImportError:parser.error("Use the CARC Python environment where your PyTorch reference runs")
        torch_version=torch.__version__
        torch_threads=torch.get_num_threads()
        reference=reference_torch
    else:reference=reference_numpy
    try:
        cases,provenance=prepare_cases(args)
        payload=encode_cases(cases,reference)
    except (ValueError,OSError,KeyError,OverflowError,ImportError) as exc:
        parser.error(str(exc))
    mode=provenance["dataset"]
    if mode=="smoke_synthetic":
        print("SMOKE TEST: generated setup inputs; these are not real-model experiment results.",flush=True)
    print(f"Checking and timing {len(cases)} cases, both CUDA samplers...",flush=True)
    completed=subprocess.run([str(args.executable.resolve()),str(args.warmup),str(args.repeats),
                              str(args.launches_per_sample)],input=payload,text=True,capture_output=True)
    if completed.returncode:
        print(completed.stderr or "CUDA benchmark driver failed",file=sys.stderr)
        print("No new benchmark results were saved. Any older output files were left unchanged.",file=sys.stderr)
        return completed.returncode if completed.returncode>0 else 1
    try:
        native_metadata,results=parse_output(completed.stdout,cases,args.repeats)
        rows,raw=summarize(cases,results,args.launches_per_sample)
    except ValueError as exc:
        print(f"Invalid native results: {exc}",file=sys.stderr)
        return 1
    output=args.output or Path("results")/("sampler_benchmark_smoke.csv" if args.smoke_test else "sampler_benchmark.csv")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="") as target:
        writer=csv.DictWriter(target,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    root=Path(__file__).resolve().parents[1]
    revision=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],text=True,capture_output=True)
    input_path=args.logits or args.input
    files={"executable":args.executable,"sampler_source":root/"src/sampler.cu",
           "sampler_header":root/"include/sampler.h","benchmark_driver":root/"benchmarks/sampler_benchmark_driver.cu",
           "python_runner":Path(__file__),"python_utilities":Path(__file__).with_name("sampler_benchmark_utils.py")}
    if input_path:files["input"]=input_path
    hashes={name:dict(path=str(path.resolve()),sha256=sha256(path)) for name,path in files.items() if path.is_file()}
    metadata=dict(scope="sampler only; selection is untimed input preparation",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        provenance=provenance,reference=args.reference,reference_precision="CPU FP64 on identical FP32 inputs",
        validation="both methods checked against Python before warmup/timing in every case; exact original-ID matches required",
        warmup_per_method=args.warmup,repeats=args.repeats,launches_per_sample=args.launches_per_sample,
        statistic="median of individual event-based launch latencies" if args.launches_per_sample==1 else "median of group-average per-launch event times",
        timing_excludes=["input preparation/Top-k","Python reference","file I/O","allocation","CPU-GPU transfers","validation","Python subprocess/text conversion"],
        timing_includes=["sampler kernel","probability/CDF scratch writes","any GPU idle gaps inside event intervals"],
        method_order="alternating serial/parallel order across repetitions",
        scratch_bytes_per_case="B*k*2*sizeof(float) for probabilities and CDF; same buffers for both methods",
        hardware=native_metadata,slurm_job_id=os.environ.get("SLURM_JOB_ID"),hostname=platform.node(),
        platform=platform.platform(),python=sys.version,numpy=np.__version__,torch=torch_version,
        torch_cpu_threads=torch_threads,git_revision=revision.stdout.strip() if revision.returncode==0 else None,
        file_hashes=hashes,raw_timings=raw)
    json_path=output.with_suffix(".json")
    json_path.write_text(json.dumps(metadata,indent=2)+"\n")
    print("B     k       T       serial_us    parallel_us    speedup")
    for index in range(0,len(rows),2):
        a,b=rows[index:index+2]
        print(f"{a['B']:<5} {a['k']:<7} {a['temperature']:<7.3g} {a['median_us']:<12.3f} {b['median_us']:<14.3f} {b['speedup_vs_serial_gpu']:.3f}x")
    print(f"Correctness: {2*len(cases)}/{2*len(cases)} method/case checks passed.")
    print(f"Saved {output} and {json_path}")
    return 0


if __name__=="__main__":sys.exit(main())
