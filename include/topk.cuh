#pragma once
#include <cuda_runtime.h>

#include <cstddef>

namespace topk {
// All buffers are device-resident. Scores/outputs are contiguous row-major.
// Order: descending finite FP32 score, then ascending original token ID.
// Padding and workspace are internal; input is never modified.
struct FullSortWorkspace {
    float* values;
    int* ids;
    std::size_t capacity;  // Number of elements available in EACH buffer.
};

// Returns zero for invalid dimensions/overflow. Allocate this many floats + ints.
std::size_t full_sort_workspace_elements(int batch, int vocab);

// Enqueues work on stream; caller owns buffers until stream completion.
// Outputs [batch,k]; workspace [batch,next_power_of_two(vocab)].
// No allocation, host transfer, or synchronization occurs in this function.
// Returns launch/argument errors; execution errors surface at synchronization.
cudaError_t full_sort(const float* input, int batch, int vocab, int k,
                      float* output_values, int* output_ids,
                      FullSortWorkspace workspace, cudaStream_t stream = nullptr);
}  // namespace topk
