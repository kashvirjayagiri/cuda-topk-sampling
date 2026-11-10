# Kashvi's full-sort baseline

## Reading order and eight-score walkthrough

Only Kashvi's full-sort component is covered here. John's hierarchical selector and Joel's sampler remain their own implementations.

Start with `full_sort()` at the bottom of `src/full_sort.cu`: it reads like a recipe. Then read the three kernels it calls. `__global__` means a function launched on the GPU; `__device__` marks the small ordering helper used by GPU threads. Ordinary functions here run on the CPU and schedule GPU work.

| Name | Responsibility |
|---|---|
| `copy_and_pad_kernel` | Copy scores, attach original IDs, add losing padding |
| `compare_pairs_kernel` | Perform one parallel comparison stage |
| `copy_top_k_kernel` | Return the best k pairs from each fully sorted row |
| `next_power_of_two` | Compute the padded row length |
| `blocks_needed` | Choose how many blocks to launch |
| `comes_before` | Higher score wins; smaller ID breaks ties |

For scores `[3, 1, 7, 4, 7, 2, 6, 5]` and k=3:

- Input IDs are `[0, 1, 2, 3, 4, 5, 6, 7]`; keep each ID attached to its score.
- Eight is a power of two, so no padding is needed.
- The network uses `(group size, comparison distance)` stages `(2,1)`, `(4,2)`, `(4,1)`, `(8,4)`, `(8,2)`, `(8,1)`.
- After group size 4 finishes, one half runs best-first and the other worst-first. This alternating order is intentional; the size-8 stages merge them into final order.
- Sorted scores are `[7, 7, 6, 5, 4, 3, 2, 1]` with IDs `[2, 4, 6, 7, 3, 0, 5, 1]`.
- Return scores `[7, 7, 6]` and IDs `[2, 4, 6]`.

Inside a stage, `column ^ partner_distance` chooses the comparison partner by flipping one position bit. For distance 1, pairs are `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`; for distance 2, they are `(0,2)`, `(1,3)`, `(4,6)`, `(5,7)`. Only the lower-index member writes each pair. Partners stay within the same row.

This example illustrates the algorithm; it does not verify CUDA compilation or GPU execution. Use the CARC tests below for that.

## What is implemented

`src/full_sort.cu` implements a complete GPU bitonic sorting network. It sorts every row, then copies its first k entries. Score/token-ID pairs remain together. Higher scores win; ties favor smaller original IDs. Finite FP32 inputs are supported.

1. Copy input into workspace and pad each row to the next power of two with negative infinity and invalid IDs.
2. For each network width and compare distance, launch a compare/exchange kernel. Each pair is updated by one thread. Launches on the same stream provide ordering between stages, including comparisons across blocks.
3. Copy the first k pairs from each row into the output buffers.

This is the straightforward required baseline. It intentionally uses global-memory stages and can incur many launches. For V=50,257, the padded size is 65,536 and there are 136 compare/exchange launches, plus initialization and output kernels. Measure rather than assuming it is fast.

## Build and test on CARC

Run in the allocated GPU session, from the repo root. Activate/load a CUDA toolkit with `nvcc` available; the PyTorch environment alone is insufficient. First inspect your modules and compiler:

```bash
nvidia-smi
nvcc --version
cmake --version
```

For the Tesla P100, use a CUDA 12.x toolkit that supports compute capability 6.0, and build for architecture 60. Do not select a CUDA 13 toolkit to compile this P100 target. Use your CARC module listing to select the available compatible toolkit; module names are environment-specific.

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

For another GPU, use its matching architecture instead of 60 and a separate build directory if changing toolkits. Tests compare GPU results with an independent CPU `std::sort` reference. They cover random values, repeated values, all-equal negative values, k=1, k=V, non-power-of-two sizes, cross-block comparisons, batches, the 50,257-entry vocabulary, and preservation of the input.

Optional memory/race diagnostic when Compute Sanitizer is available:

```bash
compute-sanitizer --tool memcheck ./build/test_full_sort
```

The test is correctness-only: allocation and CPU reference sorting are included in the test program, so its total runtime is not a selection benchmark.

## Calling the implementation

Include `include/topk.cuh` and link the `full_sort` target. The interface accepts GPU pointers, not `.pt` files. Allocate input `[B,V]`, outputs `[B,k]`, and workspace on the GPU. `full_sort_workspace_elements(B,V)` returns the number of elements to allocate in each workspace buffer (one float buffer, one int buffer). `capacity` is elements, not bytes.

```cpp
const auto count = topk::full_sort_workspace_elements(B, V);
// Allocate work_values: count * sizeof(float); work_ids: count * sizeof(int).
topk::FullSortWorkspace workspace{work_values, work_ids, count};
cudaError_t status = topk::full_sort(
    input, B, V, k, output_values, output_ids, workspace, stream);
// Check status, then synchronize stream before reading results/checking execution errors.
```

All arrays must be contiguous, have sufficient capacity, and not overlap input, outputs, or workspace. Keep buffers alive until stream completion. Outputs are descending FP32 values and int32 original token IDs. The function performs no allocations, copies to CPU, or explicit device synchronization. Input sizes and k must be positive, k <= V, and V <= 2^30; practical GPU memory limits are much smaller.

## Validate the saved DistilGPT2 inputs

The Python bridge calls the same C++ sorter through a small shared library; it does not reimplement sorting or require PyTorch C++ headers. The `FullSort` object allocates reusable outputs and workspace once, uses PyTorch's current CUDA stream, and checks dimensions/device/dtype. Keep the input and sorter alive until GPU work finishes. Use a separate sorter per concurrent stream. Outputs are overwritten on the next call.

After pulling these changes on CARC, reconfigure so CMake discovers the new shared-library target:

```bash
module load cuda/12.6.3
source ~/ee451-env/bin/activate
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

Start with one batch size and one k:

```bash
python scripts/validate_full_sort.py --batch-sizes 32 --k 10
```

This checks all available groups once per file. To limit it to the existing smoke-test group, add `--data data/logits/smoke` if that folder exists. To run the full matrix:

```bash
python scripts/validate_full_sort.py
```

For ten 32-row groups, the full matrix checks **1,480 GPU calls**: `(32 + 4 + 1) batches * 4 k values * 10 groups`. Each call checks exact token IDs against a stable CPU sort, score agreement within 1e-5 absolute/relative tolerance, and input preservation. The stable CPU ordering explicitly handles ties. Nonfinite inputs are rejected before testing.

`--library PATH` can point to a shared library in another build directory. Running `--help` needs only Python; actual validation needs CUDA-enabled PyTorch, saved inputs, and the compiled library.

## Verification status and next steps

The core C++ sorter compiled and passed the 52-case GPU test suite on the team's CARC P100. The Python bridge compiled on CARC and the saved-data validator passed all 1,480 calls across ten groups, B=1/8/32, and k=1/10/50/100, as reported by Kashvi. No performance claims are established yet.

Next: run saved-data validation, then measure selection latency with CUDA events after warm-up and report workspace usage. This change does not implement John's hierarchical selector, Joel's sampler, or end-to-end benchmarks.


## Selection-only benchmark

The native timing API still needs verification on CARC. Rebuild the shared library after pulling these changes.

```bash
python scripts/benchmark_full_sort.py --groups 01_everyday --batch-sizes 32 --k 10
```

After checking this first configuration, run all groups and configurations:

```bash
python scripts/benchmark_full_sort.py
```

Default: 10 warm-up calls and 50 timed repetitions per window, covering all rows. Optional flags: `--warmup`, `--repetitions`, `--data`, `--groups`, `--batch-sizes`, `--k`, `--library`, `--output` (must be a new directory).

Results are saved under a fresh `results/full-sort-.../` directory:

- `summary.csv`: per-window median/min/max milliseconds, rows per second, explicit input/output/workspace bytes.
- `samples.csv`: every individual CUDA-event duration.
- `metadata.json`: hardware/software, code/data hashes, settings, and completion status.

Timings are native CUDA-event intervals around the entire multi-launch selection call. Python call overhead is outside the interval; native launch-submission gaps can still contribute. These are selection-operation latency measurements, not profiler sums of individual kernel durations. Sampling, disk reads, transfers, allocation, and correctness checks are excluded. Console summaries are medians of window medians, while CSV retains the individual windows for analysis. Memory values count explicit tensors, not allocator reservation or process peak. Do not report these as end-to-end sampling results or speedups without the corresponding comparisons.
