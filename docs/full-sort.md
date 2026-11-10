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

## Verification status and next steps

The source and tests were prepared on a Mac without nvcc or an NVIDIA GPU. CUDA compilation and GPU correctness are pending a CARC run. No performance claims are established yet.

Next: run the commands above; address any compiler/test failures; add the team's Python/C++ bridge for the saved `.pt` tensors; then benchmark with CUDA events after warm-up, excluding allocations and transfers. The dataset loader, PyTorch bridge, sampler, and benchmark runner are not part of this baseline implementation.
