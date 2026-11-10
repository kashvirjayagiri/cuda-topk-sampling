#include <climits>
#include <cstddef>
#include <limits>
#include <math_constants.h>

#include "topk.cuh"

namespace topk {
namespace {

constexpr int kTileSize = 256;

__device__ bool comes_before(float score_a, int id_a, float score_b, int id_b) {
    return score_a > score_b || (score_a == score_b && id_a < id_b);
}

__global__ void local_tile_kernel(const float* input, int vocab, int tile_count,
                                  int k, float* output_values, int* output_ids) {
    const int tile = blockIdx.x % tile_count;
    const int row = blockIdx.x / tile_count;
    const int lane = threadIdx.x;
    const std::size_t token = static_cast<std::size_t>(tile) * kTileSize + lane;
    const std::size_t input_index = static_cast<std::size_t>(row) * vocab + token;

    __shared__ float scores[kTileSize];
    __shared__ int ids[kTileSize];
    scores[lane] = token < static_cast<std::size_t>(vocab) ? input[input_index] : -CUDART_INF_F;
    ids[lane] = token < static_cast<std::size_t>(vocab) ? static_cast<int>(token) : INT_MAX;
    __syncthreads();

    for (int width = 2; width <= kTileSize; width *= 2) {
        for (int distance = width / 2; distance > 0; distance /= 2) {
            const int partner = lane ^ distance;
            if (lane < partner) {
                const bool best_first = (lane & width) == 0;
                const bool partner_first = comes_before(scores[partner], ids[partner],
                                                        scores[lane], ids[lane]);
                if (partner_first == best_first) {
                    const float score = scores[lane];
                    const int id = ids[lane];
                    scores[lane] = scores[partner];
                    ids[lane] = ids[partner];
                    scores[partner] = score;
                    ids[partner] = id;
                }
            }
            __syncthreads();
        }
    }

    if (lane < k) {
        const std::size_t output_index =
            (static_cast<std::size_t>(row) * tile_count + tile) * k + lane;
        output_values[output_index] = scores[lane];
        output_ids[output_index] = ids[lane];
    }
}

}  // namespace

int local_tile_count(int vocab) {
    return vocab > 0 ? 1 + (vocab - 1) / kTileSize : 0;
}

cudaError_t local_tile_topk(const float* input, int batch, int vocab, int k,
                            float* output_values, int* output_ids, cudaStream_t stream) {
    const int tile_count = local_tile_count(vocab);
    if (batch <= 0 || tile_count == 0 || k <= 0 || k > vocab || k > kTileSize ||
        !input || !output_values || !output_ids) {
        return cudaErrorInvalidValue;
    }

    const auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    const auto tiles = static_cast<std::size_t>(batch) * tile_count;
    if (tiles > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        tiles > max_elements / static_cast<std::size_t>(k)) {
        return cudaErrorInvalidValue;
    }

    local_tile_kernel<<<static_cast<int>(tiles), kTileSize, 0, stream>>>(
        input, vocab, tile_count, k, output_values, output_ids);
    return cudaGetLastError();
}

}  // namespace topk
