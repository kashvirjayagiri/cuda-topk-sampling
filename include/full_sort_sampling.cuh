#pragma once

#include "sampler.h"
#include "topk.cuh"

namespace topk {

struct FullSortSamplingBuffers {
    FullSortWorkspace sort;
    float* selected_scores;         // [batch, k]
    std::int32_t* selected_ids;     // [batch, k]
    float* probabilities;           // [batch, k]
    float* cdf;                     // [batch, k]
    std::int32_t* sampled_ids;      // [batch]
};

// All buffers are distinct, caller-owned GPU allocations. Randoms contains
// one value in [0,1) per row. Keep buffers alive until stream completion.
inline cudaError_t full_sort_sample(
    const float* logits, const float* randoms, int batch, int vocab, int k,
    float temperature, FullSortSamplingBuffers buffers, cudaStream_t stream) {
    const auto status = full_sort(logits, batch, vocab, k, buffers.selected_scores,
                                  buffers.selected_ids, buffers.sort, stream);
    if (status != cudaSuccess) {
        return status;
    }

    // The same stream guarantees that the selection output is ready to read.
    return jjk::sample_parallel(buffers.selected_scores, buffers.selected_ids,
                                randoms, batch, k, temperature, buffers.probabilities,
                                buffers.cdf, buffers.sampled_ids, stream);
}

}  // namespace topk
