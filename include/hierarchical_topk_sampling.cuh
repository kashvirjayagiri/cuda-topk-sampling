#pragma once

#include "sampler.h"
#include "topk.cuh"

namespace topk {

struct HierarchicalTopKSamplingBuffers {
    HierarchicalWorkspace selection;
    float* selected_scores;       // [batch,k]
    std::int32_t* selected_ids;   // [batch,k]
    float* probabilities;         // [batch,k]
    float* cdf;                   // [batch,k]
    std::int32_t* sampled_ids;    // [batch]
};

// All buffers and randoms are distinct caller-owned device allocations.
// Randoms contains one value in [0,1) per row. Requires 1 <= k <= min(V,256).
// Caller retains all buffers until stream completion. No allocation, transfer,
// or synchronization occurs here; return the first launch/argument error.
inline cudaError_t hierarchical_topk_sample(
    const float* logits, const float* randoms, int batch, int vocab, int k,
    float temperature, HierarchicalTopKSamplingBuffers buffers, cudaStream_t stream) {
    const auto status = hierarchical_topk(logits, batch, vocab, k,
                                          buffers.selected_scores, buffers.selected_ids,
                                          buffers.selection, stream);
    if (status != cudaSuccess)
        return status;
    return jjk::sample_parallel(buffers.selected_scores, buffers.selected_ids,
                                randoms, batch, k, temperature, buffers.probabilities,
                                buffers.cdf, buffers.sampled_ids, stream);
}

}  // namespace topk
