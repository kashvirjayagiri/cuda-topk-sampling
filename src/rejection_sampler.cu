#include <climits>
#include <cassert>
#include <cmath>
#include <cstddef>

#include <curand_kernel.h>

#include "rejection_sampler.cuh"

namespace topk {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ bool before(float score_a, int id_a,
                                       float score_b, int id_b) {
    return score_a > score_b || (score_a == score_b && id_a < id_b);
}

// SplitMix64 is a bijection on 64-bit words. Distinct draw indices give
// distinct Philox seeds for a fixed caller seed; rows use separate subsequences.
__device__ __forceinline__ unsigned long long draw_seed(unsigned long long seed,
                                                         unsigned long long draw_index) {
    unsigned long long x = draw_index + 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return seed ^ (x ^ (x >> 31));
}

__device__ __forceinline__ double weight(float score, float maximum,
                                          float temperature) {
    return exp((static_cast<double>(score) - static_cast<double>(maximum)) /
               static_cast<double>(temperature));
}

__global__ void rejection_kernel(const float* logits, int vocab, int k,
                                 float temperature, unsigned long long seed,
                                 unsigned long long draw_index,
                                 int* sampled_ids, int* attempts) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const std::size_t base = static_cast<std::size_t>(row) * vocab;
    __shared__ float max_scores[kThreads];
    __shared__ int max_ids[kThreads];
    __shared__ double sums[kThreads];
    __shared__ int ranks[kThreads];
    __shared__ float maximum;
    __shared__ float pivot_score;
    __shared__ int pivot_id;
    __shared__ int has_pivot;
    __shared__ double target;
    __shared__ int winning_lane;
    __shared__ float candidate_score;
    __shared__ int candidate_id;
    __shared__ int done;

    float local_max = -INFINITY;
    int local_id = INT_MAX;
    for (std::size_t token = lane; token < static_cast<std::size_t>(vocab);
         token += kThreads) {
        const float score = logits[base + token];
        if (before(score, static_cast<int>(token), local_max, local_id)) {
            local_max = score;
            local_id = static_cast<int>(token);
        }
    }
    max_scores[lane] = local_max;
    max_ids[lane] = local_id;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (lane < stride && before(max_scores[lane + stride], max_ids[lane + stride],
                                    max_scores[lane], max_ids[lane])) {
            max_scores[lane] = max_scores[lane + stride];
            max_ids[lane] = max_ids[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        maximum = max_scores[0];
        pivot_score = -INFINITY;
        pivot_id = INT_MAX;
        has_pivot = 0;
        if (k == 1) {
            sampled_ids[row] = max_ids[0];
            attempts[row] = 1;
        }
    }
    __syncthreads();
    if (k == 1)
        return;  // Uniform branch: no thread remains at a barrier.

    curandStatePhilox4_32_10_t rng;
    if (lane == 0)
        curand_init(draw_seed(seed, draw_index), static_cast<unsigned long long>(row),
                    0ULL, &rng);

    int draws = 0;
    int previous_rank = vocab;
    while (true) {
        double local_sum = 0.0;
        for (std::size_t token = lane; token < static_cast<std::size_t>(vocab);
             token += kThreads) {
            const float score = logits[base + token];
            if (!has_pivot || before(score, static_cast<int>(token), pivot_score, pivot_id))
                local_sum += weight(score, maximum, temperature);
        }
        sums[lane] = local_sum;
        __syncthreads();
        // Inclusive Hillis-Steele scan of the 256 thread sums.
        for (int offset = 1; offset < kThreads; offset *= 2) {
            const double add = lane >= offset ? sums[lane - offset] : 0.0;
            __syncthreads();
            sums[lane] += add;
            __syncthreads();
        }
        if (lane == 0) {
            // curand() supplies [0, 2^32); the target is in [0, total).
            const double uniform = static_cast<double>(curand(&rng)) / 4294967296.0;
            target = uniform * sums[kThreads - 1];
            winning_lane = kThreads - 1;
        }
        __syncthreads();
        if (sums[lane] > target)
            atomicMin(&winning_lane, lane);
        __syncthreads();

        if (lane == winning_lane) {
            const double local_target = target - (lane == 0 ? 0.0 : sums[lane - 1]);
            double cumulative = 0.0;
            int last_positive = -1;
            int chosen = -1;
            for (std::size_t token = lane; token < static_cast<std::size_t>(vocab);
                 token += kThreads) {
                const float score = logits[base + token];
                if (!has_pivot || before(score, static_cast<int>(token), pivot_score, pivot_id)) {
                    const double token_weight = weight(score, maximum, temperature);
                    if (token_weight > 0.0) {
                        last_positive = static_cast<int>(token);
                        cumulative += token_weight;
                        if (chosen < 0 && cumulative > local_target)
                            chosen = static_cast<int>(token);
                    }
                }
            }
            if (chosen < 0)
                chosen = last_positive;  // Rounding-safe final boundary.
            candidate_id = chosen;
            candidate_score = logits[base + chosen];
        }
        __syncthreads();

        int local_rank = 0;
        for (std::size_t token = lane; token < static_cast<std::size_t>(vocab);
             token += kThreads)
            local_rank += before(logits[base + token], static_cast<int>(token),
                                 candidate_score, candidate_id);
        ranks[lane] = local_rank;
        __syncthreads();
        for (int stride = kThreads / 2; stride > 0; stride /= 2) {
            if (lane < stride)
                ranks[lane] += ranks[lane + stride];
            __syncthreads();
        }
        if (lane == 0) {
            ++draws;
            const int candidate_rank = ranks[0];
            // Diagnostic invariant: every rejection must improve the pivot rank.
            assert(candidate_rank < previous_rank);
            assert(draws <= vocab - k + 1);
            if (candidate_rank >= previous_rank) {
                sampled_ids[row] = -1;
                attempts[row] = -draws;
                done = 1;
            } else if (candidate_rank < k) {
                sampled_ids[row] = candidate_id;
                attempts[row] = draws;
                done = 1;
            } else {
                pivot_score = candidate_score;
                pivot_id = candidate_id;
                has_pivot = 1;
                previous_rank = candidate_rank;
                done = 0;
            }
        }
        __syncthreads();
        if (done)
            break;
    }
}

}  // namespace

cudaError_t rejection_sample_topk(const float* logits, int batch, int vocab, int k,
                                  float temperature, unsigned long long seed,
                                  unsigned long long draw_index, int* sampled_ids,
                                  int* attempts, cudaStream_t stream) {
    if (!logits || !sampled_ids || !attempts || batch <= 0 || vocab <= 0 ||
        k <= 0 || k > vocab || !std::isfinite(temperature) || temperature <= 0.0f)
        return cudaErrorInvalidValue;
    rejection_kernel<<<batch, kThreads, 0, stream>>>(
        logits, vocab, k, temperature, seed, draw_index, sampled_ids, attempts);
    return cudaGetLastError();
}

}  // namespace topk
