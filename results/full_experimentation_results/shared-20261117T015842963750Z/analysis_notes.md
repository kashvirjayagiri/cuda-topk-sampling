# Shared comparison: measurement notes

GPU: Tesla V100-PCIE-32GB. Timing backend: graph.
Validated method/case rows: 5920. Repetitions per scope: 101.

All four methods use the same logits, FP32 precision, temperature, supplied random values, CUDA stream, and compiled parallel sampler. Selection and full pipeline are timed separately; pipeline time is measured directly, not computed by adding medians.

`summary.csv` retains each dataset and row window. `comparisons.csv` adds paired speedups on identical configurations. `aggregate.csv` reports the median of window medians, separately by real/synthetic dataset kind, score distribution, vocabulary size, B, k, and temperature. Paired speedups are aggregated by their own median, not by dividing aggregate medians.

Speedup = baseline latency / method latency. Values above 1 mean the method is faster. The PyTorch baseline includes tie repair/order and int32 conversion. Its raw Top-k diagnostic column does not satisfy the full ordering contract and is not used for those speedups. An advantage over this adapter is not automatically an advantage over bare torch.topk.

Random policy: Uniform FP32 draws conditioned away from the union of CDF boundaries; CDF margin=5e-05. The exact values are in metadata.json. Boundary-conditioned inputs support robust numerical comparison, not claims about the statistical quality of random sampling.

CUDA graph replay measurements exclude capture/setup and reduce host-dispatch differences. Eager measurements include API dispatch gaps and may include internal PyTorch workspace allocator costs. Do not combine the two timing modes in a comparison. Explicit workspace bytes exclude graph pools, library-internal temporaries, and allocator/process peaks. Radix workspace is the queried byte count, including CUB scratch, allocated before capture or timing. Graph and eager runs must be reported separately.

Topic groups and row windows are different workloads, not independent timing repetitions. Plot shading shows workload variation, not a confidence interval. The real dataset is a controlled DistilGPT2 sample, not evidence about all models or text inputs.

## Observed paired wins

- hierarchical was faster than full_sort for selection in 1480/1480 tested configurations/windows.
- hierarchical was faster than full_sort for pipeline in 1480/1480 tested configurations/windows.
- hierarchical was faster than pytorch for selection in 1480/1480 tested configurations/windows.
- hierarchical was faster than pytorch for pipeline in 1480/1480 tested configurations/windows.
- radix was faster than full_sort for selection in 1480/1480 tested configurations/windows.
- radix was faster than full_sort for pipeline in 1480/1480 tested configurations/windows.
- radix was faster than pytorch for selection in 1480/1480 tested configurations/windows.
- radix was faster than pytorch for pipeline in 1480/1480 tested configurations/windows.
- pytorch was faster than full_sort for selection in 1480/1480 tested configurations/windows.
- pytorch was faster than full_sort for pipeline in 1480/1480 tested configurations/windows.

Use the CSVs/figures to discuss where performance changes with B, k, vocabulary size, and distribution. Avoid treating the win counts alone as an overall performance score.
