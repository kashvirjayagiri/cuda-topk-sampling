"""input prep cpu refs and output parse for sampler benchmark"""
import math
from pathlib import Path
import numpy as np


def validate_case(case):
    raw_ids = np.asarray(case["ids"])
    if not np.issubdtype(raw_ids.dtype, np.integer):
        raise ValueError("Token IDs must be integers")
    if raw_ids.size and (raw_ids.min()<0 or raw_ids.max()>np.iinfo(np.int32).max):
        raise ValueError("Token IDs must fit nonnegative int32")
    with np.errstate(over="ignore",under="ignore",invalid="ignore"):
        s=np.ascontiguousarray(case["scores"],dtype=np.float32)
        u=np.ascontiguousarray(case["randoms"],dtype=np.float32)
        t=float(np.float32(case["temperature"]))
    ids=np.ascontiguousarray(raw_ids,dtype=np.int32)
    if s.ndim!=2 or min(s.shape)<1 or s.size>16777216:
        raise ValueError("Scores must have nonempty shape [B,k], at most 16,777,216 elements")
    if ids.shape!=s.shape or u.shape!=(s.shape[0],):
        raise ValueError("Require scores/IDs [B,k], random values [B]")
    if not np.isfinite(s).all() or not np.isfinite(t) or t<=0:
        raise ValueError("Scores must be finite FP32 and temperature positive finite FP32")
    if not np.isfinite(u).all() or np.any(u<0) or np.any(u>=1):
        raise ValueError("Random values must lie in [0,1) after FP32 conversion")
    if np.any(s[:,1:]>s[:,:-1]): raise ValueError("Selected scores must be descending")
    if np.any((s[:,1:]==s[:,:-1]) & (ids[:,1:]<=ids[:,:-1])):
        raise ValueError("Equal scores must have ascending original IDs")
    if any(np.unique(row).size!=row.size for row in ids):
        raise ValueError("Token IDs must be distinct within each row")
    return {**case,"scores":s,"ids":ids,"randoms":u,"temperature":t}


def reference_numpy(case):
    s=case["scores"].astype(np.float64)
    shifted=s-s.max(axis=1,keepdims=True)
    weights=np.exp(shifted/case["temperature"])
    p=weights/weights.sum(axis=1,keepdims=True)
    cdf=np.minimum(np.cumsum(p,axis=1),1.)
    cdf[:,-1]=1.
    positions=np.sum(cdf<=case["randoms"][:,None],axis=1)
    chosen=case["ids"][np.arange(len(s)),positions]
    return p,cdf,chosen


def reference_torch(case):
    import torch
    with torch.no_grad():
        s=torch.tensor(case["scores"],dtype=torch.float64,device="cpu")
        ids=torch.tensor(case["ids"],dtype=torch.int32,device="cpu")
        u=torch.tensor(case["randoms"],dtype=torch.float64,device="cpu")
        shifted=s-s.max(dim=1,keepdim=True).values
        p=torch.softmax(shifted/case["temperature"],dim=1)
        cdf=torch.cumsum(p,dim=1).clamp(max=1.)
        cdf[:,-1]=1.
        positions=torch.searchsorted(cdf,u[:,None].contiguous(),right=True)
        chosen=ids.gather(1,positions).squeeze(1)
        return p.numpy(),cdf.numpy(),chosen.numpy()


def load_logits(path):
    path=Path(path)
    if path.suffix in (".pt",".pth"):
        import torch
        loaded=torch.load(path,map_location="cpu",weights_only=True)
        if isinstance(loaded,dict):
            if "logits" not in loaded: raise ValueError("Tensor dictionary must contain 'logits'")
            loaded=loaded["logits"]
        if not isinstance(loaded,torch.Tensor): raise ValueError("Expected a logits tensor")
        data=loaded.detach().to(torch.float32).cpu().numpy()
    else:
        loaded=np.load(path,allow_pickle=False,mmap_mode="r")
        if isinstance(loaded,np.lib.npyio.NpzFile):
            with loaded:
                if "logits" in loaded.files: data=loaded["logits"].copy()
                elif len(loaded.files)==1: data=loaded[loaded.files[0]].copy()
                else: raise ValueError("NPZ must contain 'logits' or one unambiguous logits array")
        else: data=loaded
    with np.errstate(over="ignore",invalid="ignore"):
        data=np.asarray(data,dtype=np.float32)
    if data.ndim!=2 or min(data.shape)<1 or not np.isfinite(data).all():
        raise ValueError("Raw logits must be a finite FP32-convertible [N,V] matrix")
    if data.shape[1]>np.iinfo(np.int32).max:
        raise ValueError("Vocabulary positions must fit int32")
    return data


def prepare_cases(args):
    rng=np.random.default_rng(args.seed)
    if args.windows<1 or args.start_row<0: raise ValueError("Invalid window count/start row")
    temperatures=args.temperatures
    if not temperatures or any(not np.isfinite(t) or t<=0 for t in temperatures):
        raise ValueError("Temperatures must be finite and positive")
    supplied_randoms=None
    original_ids=None
    if args.input:
        with np.load(args.input,allow_pickle=False) as data:
            required={"scores","token_ids","random_values"}
            if not required.issubset(data.files):
                raise ValueError("Selected-input NPZ requires scores, token_ids, random_values")
            raw=data["scores"].copy()
            original_ids=data["token_ids"].copy()
            supplied_randoms=data["random_values"].copy()
        if raw.ndim!=2 or min(raw.shape)<1:
            raise ValueError("Selected scores must have nonempty shape [N,K]")
        batch_sizes=args.batch_sizes or [raw.shape[0]]
        k_values=args.k_values or [raw.shape[1]]
        # validate full supplied rows before slices
        validate_case(dict(scores=raw,ids=original_ids,randoms=supplied_randoms,temperature=1.))
        mode="selected_file"
        vocabulary=None
    elif args.logits:
        raw=load_logits(args.logits)
        batch_sizes=args.batch_sizes or [1,8,32]
        k_values=args.k_values or [1,10,50,100]
        mode="logits_file"
        vocabulary=int(raw.shape[1])
    else:
        batch_sizes=args.batch_sizes or [1,8,32]
        k_values=args.k_values or [1,10,50,100]
        if min(batch_sizes)<1 or min(k_values)<1: raise ValueError("B and k must be positive")
        if max(batch_sizes)*max(k_values)>16777216:
            raise ValueError("Requested B*k exceeds benchmark driver limit")
        # smoke data only
        raw=rng.normal(size=(args.start_row+max(batch_sizes)*args.windows,max(k_values))).astype(np.float32)
        mode="smoke_synthetic"
        vocabulary=None
    if any(B<1 for B in batch_sizes) or any(k<1 for k in k_values):
        raise ValueError("B and k must be positive")
    max_B,max_k=max(batch_sizes),max(k_values)
    if args.start_row+max_B*args.windows>raw.shape[0]:
        raise ValueError(f"Need at least {args.start_row+max_B*args.windows} rows; input has {raw.shape[0]}. Set --batch-sizes and/or --windows accordingly.")
    if max_k>raw.shape[1]: raise ValueError(f"Requested k={max_k} exceeds available width={raw.shape[1]}")
    cases=[]
    for window in range(args.windows):
        start=args.start_row+window*max_B
        rows=np.asarray(raw[start:start+max_B],dtype=np.float32)
        if original_ids is None:
            # cpu topk prep outside timing
            order=np.argsort(-rows,axis=1,kind="stable")[:,:max_k]
            top_scores=np.take_along_axis(rows,order,axis=1).copy()
            top_ids=order.astype(np.int32)
        else:
            top_scores=rows[:,:max_k].copy()
            top_ids=original_ids[start:start+max_B,:max_k].copy()
        for B in batch_sizes:
            for k in k_values:
                s=top_scores[:B,:k].copy()
                ids=top_ids[:B,:k].copy()
                u=(supplied_randoms[start:start+B].copy() if supplied_randoms is not None
                   else rng.random(B,dtype=np.float32))
                for temperature in temperatures:
                    case=dict(name=f"w{window}_B{B}_k{k}_T{temperature}",scores=s,ids=ids,
                              randoms=u.copy(),temperature=temperature,window=window,
                              source_start_row=start,source_vocabulary_size=vocabulary,dataset=mode)
                    case=validate_case(case)
                    if mode=="smoke_synthetic":
                        # setup check only
                        # keep supplied randoms for file input
                        p,cdf,_=reference_numpy(case)
                        indices=np.arange(B)%min(k,10)
                        case["randoms"]=(cdf[np.arange(B),indices]-p[np.arange(B),indices]/2).astype(np.float32)
                        case=validate_case(case)
                    cases.append(case)
    provenance={"dataset":mode,"source_shape":list(raw.shape),"windows":args.windows,
                "batch_sizes":batch_sizes,"k_values":k_values,"temperatures":temperatures,
                "seed":args.seed,"source_start_row":args.start_row,
                "selection_preparation":"stable CPU reference Top-k, untimed" if original_ids is None else "supplied preselected candidates, untimed",
                "random_inputs":"supplied NPZ values" if supplied_randoms is not None else
                    ("interval midpoints for setup checks" if mode=="smoke_synthetic" else "seeded FP32 uniform values")}
    return cases,provenance


def encode_cases(cases,reference):
    lines=[]
    for case in cases:
        p,cdf,chosen=reference(case)
        B,k=case["scores"].shape
        lines.append(f"{B} {k} {case['temperature']:.17g}")
        for array in (case["scores"],case["ids"],case["randoms"],p,cdf,chosen):
            integer=np.issubdtype(np.asarray(array).dtype,np.integer)
            lines.append(" ".join(str(int(x)) if integer else format(float(x),".17g") for x in np.asarray(array).ravel()))
    return "\n".join(lines)+"\n"


def parse_output(stdout,cases,repeats):
    lines=iter(stdout.splitlines())
    metadata={}
    results=[]
    try:
        for key in ("DEVICE","COMPUTE","CUDA_RUNTIME","CUDA_DRIVER","NVCC"):
            parts=next(lines).split(" ",1)
            if len(parts)!=2 or parts[0]!=key: raise ValueError(f"Missing native {key} metadata")
            metadata[key.lower()]=parts[1]
        for index,case in enumerate(cases):
            header=next(lines).split()
            B,k=case["scores"].shape
            if len(header)!=5 or header[0]!="CASE" or [int(x) for x in header[1:4]]!=[index,B,k] or not math.isclose(float(header[4]),case["temperature"],rel_tol=1e-7):
                raise ValueError("Native case metadata does not match prepared inputs")
            modes={}
            for mode in ("serial","parallel"):
                check=next(lines).split()
                if len(check)!=5 or check[:3]!=["CHECK",mode,"PASS"]:
                    raise ValueError("Native correctness check was not a pass")
                errors=[float(x) for x in check[3:]]
                if any(not math.isfinite(x) or x<0 for x in errors): raise ValueError("Invalid validation error values")
                times=next(lines).split()
                if times[:3]!=["TIMES",mode,str(repeats)] or len(times)!=3+repeats:
                    raise ValueError("Native timing sample count does not match request")
                values=np.asarray([float(x) for x in times[3:]],dtype=np.float64)
                if not np.isfinite(values).all() or np.any(values<=0): raise ValueError("Invalid timing sample")
                modes[mode]={"times_us":values,"max_probability_error":errors[0],"max_cdf_error":errors[1]}
            if next(lines).strip()!="END": raise ValueError("Missing native case terminator")
            results.append(modes)
    except (StopIteration,IndexError) as exc:
        raise ValueError("Native output is incomplete") from exc
    if any(line.strip() for line in lines): raise ValueError("Unexpected extra native output")
    return metadata,results


def summarize(cases,results,launches_per_sample):
    rows=[]
    raw=[]
    for index,(case,result) in enumerate(zip(cases,results)):
        B,k=case["scores"].shape
        serial_median=float(np.median(result["serial"]["times_us"]))
        for mode in ("serial","parallel"):
            record=result[mode]
            values=record["times_us"]
            median=float(np.median(values))
            rows.append(dict(case=index,dataset=case["dataset"],window=case["window"],
                B=B,k=k,source_vocabulary_size=case["source_vocabulary_size"],
                temperature=case["temperature"],method=f"cuda_{mode}",correctness="passed",
                median_us=median,p10_us=float(np.percentile(values,10)),p90_us=float(np.percentile(values,90)),
                rows_per_second=B*1e6/median,speedup_vs_serial_gpu=serial_median/median,
                launches_per_sample=launches_per_sample,
                max_probability_error=record["max_probability_error"],max_cdf_error=record["max_cdf_error"]))
            raw.append(dict(case=index,method=f"cuda_{mode}",times_us=values.tolist()))
    return rows,raw
