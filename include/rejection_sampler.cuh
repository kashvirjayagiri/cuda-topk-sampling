#pragma once

#include <cuda_runtime.h>

namespace topk {

// Samples one original token ID per row from the exact tie-broken Top-k set.
// Input is contiguous finite FP32 [batch,vocab]; sampled_ids and attempts are
// preallocated int32 [batch]. attempts counts weighted draws, including the
// accepted draw. Reusing (seed, draw_index) reproduces the same result.
// Enqueues on stream without allocation, host transfer, or synchronization.
cudaError_t rejection_sample_topk(const float* logits, int batch, int vocab, int k,
                                  float temperature, unsigned long long seed,
                                  unsigned long long draw_index, int* sampled_ids,
                                  int* attempts, cudaStream_t stream = nullptr);

}  // namespace topk
