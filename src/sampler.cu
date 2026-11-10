#include "sampler.h"
#include <cmath>
#include <cstddef>

//DONE BY JOEL SEDILLO

namespace {
// fixed block size, as in the lectures' explicitly configured thread blocks
const int THREADS = 256;

// subtract BEFORE dividing, preventing overflow of large positive logits/T
// double subtraction also handles finite FP32 values at opposite extremes
__device__ float weight(float score, float maximum, float temperature) {
    double shifted_score = (double)score - (double)maximum;
    float exponent = (float)(shifted_score / (double)temperature);
    return expf(exponent);
}

// a deliberately serial GPU baseline: ONE thread processes ALL rows
__global__ void serial_kernel(const float* scores, const std::int32_t* ids,
    const float* randoms, int B, int k, float temperature,
    float* probabilities, float* cdf, std::int32_t* sampled_ids) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int row = 0; row < B; ++row) {
        // row-major address = row * row_width + column.
        const std::size_t base = (std::size_t)row * k;
        float maximum = -INFINITY;
        for (int i = 0; i < k; ++i) {
            maximum = fmaxf(maximum, scores[base+i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            probabilities[base+i] = weight(scores[base+i], maximum, temperature);
            sum += probabilities[base+i];
        }
        float cumulative = 0.0f;
        int selected = k - 1; // valid fallback for rounding at the final boundary
        bool found = false;
        for (int i = 0; i < k; ++i) {
            probabilities[base+i] /= sum;
            cumulative = fminf(1.0f, cumulative + probabilities[base+i]);
            cdf[base+i] = (i == k-1) ? 1.0f : cumulative;
            if (!found && cdf[base+i] > randoms[row]) {
                selected = i;
                found = true;
            }
        }
        sampled_ids[row] = ids[base+selected];
    }
}

// one block per row; threads cooperate on max, sum, and inclusive scan
// k need not be a power of two or <=256: each thread handles strided values
// then the block scans 256-wide chunks with a shared carry between chunks
__global__ void parallel_kernel(const float* scores, const std::int32_t* ids,
    const float* randoms, int k, float temperature,
    float* probabilities, float* cdf, std::int32_t* sampled_ids) {
    __shared__ float scratch[THREADS];
    __shared__ float maximum;
    __shared__ float total;
    __shared__ float carry;
    __shared__ int candidate_indices[THREADS];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const std::size_t base = (std::size_t)row * k;
    int local_selected = k - 1;

    float local_max = -INFINITY;
    for (int i = tid; i < k; i += THREADS) {
        local_max = fmaxf(local_max, scores[base+i]);
    }
        
    scratch[tid] = local_max;
   // synchronize before reading another thread's data

    __syncthreads();

    for (int stride = THREADS/2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid+stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        maximum = scratch[0];
    }

    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < k; i += THREADS) {
        float w = weight(scores[base+i], maximum, temperature);
        probabilities[base+i] = w;
        local_sum += w;
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = THREADS/2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid+stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        total = scratch[0];
        carry = 0.0f;
    }
    __syncthreads();

    for (int start = 0; start < k; start += THREADS) {
        const int i = start + tid;
        float p = 0.0f;
        if (i < k) {
            p = probabilities[base+i] / total;
            probabilities[base+i] = p;
        }
        // invalid lanes contribute zero and still participate in every barrier
        scratch[tid] = p;
        __syncthreads();

        // hillis-steele inclusive scan. Read, barrier, write, barrier prevents
        // a lane from overwriting a value that another lane is still reading
        for (int offset = 1; offset < THREADS; offset *= 2) {
            float add = (tid >= offset) ? scratch[tid-offset] : 0.0f;
            __syncthreads();
            scratch[tid] += add;
            __syncthreads();
        }
        if (i < k) {
            float cumulative = fminf(1.0f, carry + scratch[tid]);
            if (i == k-1) {
                cumulative = 1.0f;
            }

            cdf[base+i] = cumulative;

            if (cumulative > randoms[row] && i < local_selected) {
                local_selected = i;
            }
        }
        // every lane must finish reading carry before lane 0 updates it
        __syncthreads();
        if (tid == 0) {
            carry = fminf(1.0f, carry + scratch[THREADS-1]);
        }
        __syncthreads();
    }
    // find the smallest qualifying index with the SAME shared-memory/barrier
    // pattern used for the max and sum. No atomics or warp intrinsics needed
    candidate_indices[tid] = local_selected;
    __syncthreads();
    for (int stride = THREADS/2; stride > 0; stride /= 2) {
        if (tid < stride && candidate_indices[tid+stride] < candidate_indices[tid]) {
            candidate_indices[tid] = candidate_indices[tid+stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        sampled_ids[row] = ids[base+candidate_indices[0]];
    }
}

bool invalid(const float* scores, const std::int32_t* ids, const float* randoms,
    int B, int k, float temperature, float* probabilities, float* cdf,
    std::int32_t* sampled_ids) {
    return !scores || !ids || !randoms || !probabilities || !cdf || !sampled_ids
        || B < 1 || k < 1 || !std::isfinite(temperature) || temperature <= 0.0f;
}
}

namespace jjk {
cudaError_t sample_serial(const float* scores, const std::int32_t* ids,
    const float* randoms, int B, int k, float temperature,
    float* probabilities, float* cdf, std::int32_t* sampled_ids,
    cudaStream_t stream) {
    if (invalid(scores,ids,randoms,B,k,temperature,probabilities,cdf,sampled_ids)) {
        return cudaErrorInvalidValue;
    }
    serial_kernel<<<1,1,0,stream>>>(scores,ids,randoms,B,k,temperature,
                                  probabilities,cdf,sampled_ids);
    return cudaGetLastError();
}
cudaError_t sample_parallel(const float* scores, const std::int32_t* ids,
    const float* randoms, int B, int k, float temperature,
    float* probabilities, float* cdf, std::int32_t* sampled_ids,
    cudaStream_t stream) {
    if (invalid(scores,ids,randoms,B,k,temperature,probabilities,cdf,sampled_ids)) {
        return cudaErrorInvalidValue;
    }
    parallel_kernel<<<B,THREADS,0,stream>>>(scores,ids,randoms,k,temperature,
                                          probabilities,cdf,sampled_ids);
    return cudaGetLastError();
}
}
