"""Compare your serial/parallel CUDA sampler to PyTorch on identical inputs.

Run from the repository root after compiling sampler_compare_driver.
Text I/O and CPU/GPU copies are intentional: this is NOT a timing benchmark.
"""

# DONE BY JOEL SEDILLO

import argparse
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def reference_math(scores, ids, randoms, temperature):
    """Independent CPU answer key used to test the checker without PyTorch."""
    probabilities, cdf, chosen = [], [], []
    for values, tokens, u in zip(scores, ids, randoms):
        maximum = max(values)
        weights = [math.exp((s-maximum)/temperature) for s in values]
        denominator = math.fsum(weights)
        p = [w/denominator for w in weights]
        cumulative = [min(math.fsum(p[:i+1]), 1.) for i in range(len(p))]
        cumulative[-1] = 1.
        position = next(i for i, value in enumerate(cumulative) if value > u)
        probabilities.append(p)
        cdf.append(cumulative)
        chosen.append(tokens[position])
    return probabilities, cdf, chosen


def reference_torch(scores, ids, randoms, temperature):
    import torch
    with torch.no_grad():
        # Same FP32 inputs as CUDA; FP64 intermediates form an accurate answer key.
        s = torch.tensor(scores, dtype=torch.float64, device="cpu")
        tokens = torch.tensor(ids, dtype=torch.int32, device="cpu")
        u = torch.tensor(randoms, dtype=torch.float64, device="cpu")
        shifted = s - s.max(dim=1, keepdim=True).values
        p = torch.softmax(shifted/temperature, dim=1)
        cdf = torch.cumsum(p, dim=1).clamp(max=1.)
        cdf[:, -1] = 1.
        positions = torch.searchsorted(cdf, u[:, None].contiguous(), right=True)
        sampled = tokens.gather(1, positions).squeeze(1)
        return p.tolist(), cdf.tolist(), sampled.tolist()


def make_cases():
    rng = random.Random(451)
    cases = [
        ("worked_example", [[3.,2.,1.]], [[17,42,90]], [.8], 1.),
        ("worked_batch", [[3.,2.,1.],[0.,0.,0.]], [[17,42,90],[5,6,7]], [.8,.5], 1.),
        ("one_candidate", [[4.],[-9.],[0.]], [[7],[90],[2]], [0.,.5,fp32(1.-2**-24)], 1.),
        ("exact_uniform_boundary", [[0.,0.],[0.,0.]], [[4,9],[4,9]], [0.,.5], 1.),
        ("large_scores_small_temperature", [[1e30,0.,-1e30]], [[10,20,30]], [.5], 1e-20),
    ]
    for B in (1,8,32):
        for k in (1,10,50,100,257,513):
            for temperature in (.5,1.,2.):
                raw = [[fp32(rng.gauss(0.,1.)) for _ in range(k)] for _ in range(B)]
                order = [sorted(range(k), key=lambda i:(-row[i],i)) for row in raw]
                scores = [[row[i] for i in indices] for row,indices in zip(raw,order)]
                ids = order
                p,cdf,_ = reference_math(scores,ids,[0.]*B,temperature)
                # Midpoints in non-tiny intervals avoid ambiguous rounding boundaries.
                randoms = []
                for row in range(B):
                    index = rng.randrange(min(k,10))
                    randoms.append(fp32(cdf[row][index]-p[row][index]/2))
                cases.append((f"normal_B{B}_k{k}_T{temperature}",scores,ids,randoms,temperature))
    for name, values in (
        ("uniform", [0.]*100),
        ("nearly_uniform", [(100-i)*1e-5 for i in range(100)]),
        ("peaked", [100.]+[0.]*99),
        ("tie_heavy", [float(10-i//10) for i in range(100)]),
    ):
        scores = [values[:] for _ in range(8)]
        ids = [list(range(100)) for _ in range(8)]
        p,cdf,_ = reference_math(scores,ids,[0.]*8,1.)
        positions = [0 if name=="peaked" else row for row in range(8)]
        randoms = [fp32(cdf[row][index]-p[row][index]/2) for row,index in enumerate(positions)]
        cases.append((name,scores,ids,randoms,1.))
    return [normalize_case(case) for case in cases]


def normalize_case(case):
    name,scores,ids,randoms,t = case
    scores = [[fp32(s) for s in row] for row in scores]
    randoms = [fp32(u) for u in randoms]
    t = fp32(t)
    if not scores or not scores[0] or len(scores)!=len(ids) or len(scores)!=len(randoms):
        raise ValueError("Inputs must have nonempty shapes scores/IDs [B,k], randoms [B]")
    k = len(scores[0])
    if not math.isfinite(t) or t <= 0:
        raise ValueError("Temperature must be positive and finite in FP32")
    for row,tokens,u in zip(scores,ids,randoms):
        if len(row)!=k or len(tokens)!=k or not all(math.isfinite(s) for s in row):
            raise ValueError("Invalid score matrix")
        if not math.isfinite(u) or not 0 <= u < 1:
            raise ValueError("Random values must be in [0,1) in FP32")
        if any(type(i) is not int or i<0 or i>2**31-1 for i in tokens) or len(set(tokens))!=k:
            raise ValueError("Original IDs must be distinct nonnegative int32 values per row")
        if any(row[i]<row[i+1] or (row[i]==row[i+1] and tokens[i]>=tokens[i+1]) for i in range(k-1)):
            raise ValueError("Candidates must be ordered by descending score, then ascending ID")
    return name,scores,ids,randoms,t


def encode_cases(cases):
    lines = []
    for _,scores,ids,randoms,t in cases:
        lines.append(f"{len(scores)} {len(scores[0])} {t:.17g}")
        lines.append(" ".join(format(s,".17g") for row in scores for s in row))
        lines.append(" ".join(str(i) for row in ids for i in row))
        lines.append(" ".join(format(u,".17g") for u in randoms))
    return "\n".join(lines)+"\n"


def parse_outputs(stdout, cases):
    lines = iter(stdout.splitlines())
    results = []
    try:
        for _,scores,_,_,_ in cases:
            B,k = len(scores),len(scores[0])
            case_result = {}
            for mode in ("serial","parallel"):
                if next(lines).strip()!=mode:
                    raise ValueError("Unexpected native output mode")
                p = [float(x) for x in next(lines).split()]
                cdf = [float(x) for x in next(lines).split()]
                sampled = [int(x) for x in next(lines).split()]
                if len(p)!=B*k or len(cdf)!=B*k or len(sampled)!=B:
                    raise ValueError("Native output dimensions do not match input")
                case_result[mode] = ([p[i*k:(i+1)*k] for i in range(B)],
                                     [cdf[i*k:(i+1)*k] for i in range(B)],sampled)
            if next(lines).strip()!="END": raise ValueError("Missing case terminator")
            results.append(case_result)
    except StopIteration as exc:
        raise ValueError("Incomplete native output") from exc
    if any(line.strip() for line in lines): raise ValueError("Unexpected extra native output")
    return results


def check_result(expected, actual, ids, randoms):
    expected_p,expected_cdf,expected_ids = expected
    p,cdf,sampled = actual
    max_p_error,max_cdf_error = 0.,0.
    for row in range(len(ids)):
        if not math.isclose(math.fsum(p[row]),1.,rel_tol=1e-5,abs_tol=1e-5):
            raise AssertionError(f"row {row}: probabilities do not sum to 1")
        for i in range(len(ids[row])):
            for label,value,target in (("probability",p[row][i],expected_p[row][i]),
                                       ("CDF",cdf[row][i],expected_cdf[row][i])):
                if not math.isfinite(value) or not 0<=value<=1 or not math.isclose(value,target,rel_tol=1e-5,abs_tol=1e-5):
                    raise AssertionError(f"row {row}, candidate {i}: {label}={value}, Python={target}")
            if i and cdf[row][i]<cdf[row][i-1]-1e-7:
                raise AssertionError(f"row {row}: CDF is not monotonic")
            max_p_error=max(max_p_error,abs(p[row][i]-expected_p[row][i]))
            max_cdf_error=max(max_cdf_error,abs(cdf[row][i]-expected_cdf[row][i]))
        if cdf[row][-1]!=1.: raise AssertionError("Final CDF boundary must equal 1")
        position=next(i for i,value in enumerate(cdf[row]) if value>randoms[row])
        if sampled[row]!=ids[row][position]: raise AssertionError("GPU ID is inconsistent with its CDF")
        if sampled[row]!=expected_ids[row]:
            distance=min(abs(value-randoms[row]) for value in expected_cdf[row])
            raise AssertionError(f"row {row}: token={sampled[row]}, Python={expected_ids[row]}; "
                                 f"distance to reference boundary={distance:.3g}. Inspect rounding; this is not an exact match.")
    return max_p_error,max_cdf_error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable",type=Path,default=Path("build/sampler_compare_driver"))
    parser.add_argument("--input",type=Path,help="Optional real selected-candidate NPZ with scores, token_ids, random_values")
    parser.add_argument("--temperature",type=float,default=1.)
    parser.add_argument("--reference",choices=("torch","math"),default="torch")
    parser.add_argument("--report",type=Path,default=Path("results/sampler_comparison.json"))
    args=parser.parse_args()
    if not args.executable.is_file(): parser.error("Compile build/sampler_compare_driver first; see README")
    if args.reference=="torch":
        try: import torch
        except ImportError: parser.error("PyTorch is missing in this CARC Python environment")
        reference=reference_torch
        torch_version=torch.__version__
    else:
        reference=reference_math
        torch_version=None
    if args.input:
        import numpy as np
        with np.load(args.input,allow_pickle=False) as data:
            cases=[normalize_case((str(args.input),data["scores"].tolist(),
                   data["token_ids"].tolist(),data["random_values"].tolist(),args.temperature))]
    else:
        cases=make_cases()
    print(f"Running {len(cases)} cases using both CUDA samplers...",flush=True)
    completed=subprocess.run([str(args.executable.resolve())],input=encode_cases(cases),
                             text=True,capture_output=True)
    if completed.returncode:
        print(completed.stderr or "Native driver failed",file=sys.stderr)
        return 1
    outputs=parse_outputs(completed.stdout,cases)
    report={"reference":args.reference,"reference_precision":"CPU FP64 on identical FP32 inputs",
            "torch_version":torch_version,"atol":1e-5,"rtol":1e-5,"results":[]}
    failed=0
    for case,output in zip(cases,outputs):
        name,scores,ids,u,t=case
        expected=reference(scores,ids,u,t)
        for mode in ("serial","parallel"):
            row={"case":name,"mode":mode,"B":len(scores),"k":len(scores[0]),"temperature":t}
            try:
                p_error,cdf_error=check_result(expected,output[mode],ids,u)
                row.update(status="passed",max_probability_error=p_error,max_cdf_error=cdf_error)
            except AssertionError as exc:
                failed+=1
                row.update(status="failed",message=str(exc))
                print(f"FAIL {name} / {mode}: {exc}")
            report["results"].append(row)
    report.update(status="failed" if failed else "passed",cases=len(cases),checks=2*len(cases),failures=failed)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(f"{2*len(cases)-failed}/{2*len(cases)} checks passed against {args.reference}.")
    print(f"Report: {args.report}")
    return 1 if failed else 0


if __name__=="__main__":
    sys.exit(main())
