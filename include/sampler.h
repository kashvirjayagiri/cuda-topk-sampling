#pragma once

#include <cuda_runtime.h>
#include <cstdint>

//DONE BY JOEL SEDILLO

namespace jjk {

cudaError_t sample_serial(
    const float* scores,
    const std::int32_t* ids,
    const float* randoms,
    int B,
    int k,
    float temperature,
    float* probabilities,
    float* cdf,
    std::int32_t* sampled_ids,
    cudaStream_t stream = 0
);

cudaError_t sample_parallel(
    const float* scores,
    const std::int32_t* ids,
    const float* randoms,
    int B,
    int k,
    float temperature,
    float* probabilities,
    float* cdf,
    std::int32_t* sampled_ids,
    cudaStream_t stream = 0
);

}