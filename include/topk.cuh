#pragma once
#include <cuda_runtime.h>

#include <cstddef>

namespace topk {
// All buffers are device-resident. Scores/outputs are contiguous row-major.
// Order: descending finite FP32 score, then ascending original token ID.
// Input is never modified.
struct FullSortWorkspace {
    float* values;
    int* ids;
    std::size_t capacity;  // Number of elements available in EACH buffer.
};

// Returns zero for invalid dimensions/overflow. Allocate this many floats + ints.
std::size_t full_sort_workspace_elements(int batch, int vocab);

// Enqueues work on stream; caller owns buffers until stream completion.
// Outputs [batch,k]; workspace [batch,next_power_of_two(vocab)].
// Padding and workspace are internal to this full-sort selector.
// No allocation, host transfer, or synchronization occurs in this function.
// Returns launch/argument errors; execution errors surface at synchronization.
cudaError_t full_sort(const float* input, int batch, int vocab, int k,
                      float* output_values, int* output_ids,
                      FullSortWorkspace workspace, cudaStream_t stream = nullptr);

// Fixed 256-token tiles. Returns zero for an invalid vocabulary size.
int local_tile_count(int vocab);

// Enqueues one block per (row, tile) on stream. Input is contiguous [batch,vocab].
// Preallocated outputs are contiguous [batch,local_tile_count(vocab),k], indexed
// by ((row * tile_count + tile) * k + rank). Unused ranks contain (-inf, INT_MAX).
// Requires finite FP32 input and 1 <= k <= min(vocab,256). No allocation,
// host transfer, or synchronization occurs here. Execution errors surface at sync.
cudaError_t local_tile_topk(const float* input, int batch, int vocab, int k,
                            float* output_values, int* output_ids,
                            cudaStream_t stream = nullptr);
}  // namespace topk
