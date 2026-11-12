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

__global__ void merge_pairs_kernel(const float* input_values, const int* input_ids,
                                   int list_count, int next_count, int k, int sort_length,
                                   float* output_values, int* output_ids) {
    const int pair = blockIdx.x % next_count;
    const int row = blockIdx.x / next_count;
    const int lane = threadIdx.x;
    const int first_list = 2 * pair;
    const int second_list = first_list + 1;

    __shared__ float scores[2 * kTileSize];
    __shared__ int ids[2 * kTileSize];
    for (int offset = lane; offset < sort_length; offset += kTileSize) {
        const int list = offset < k ? first_list : second_list;
        if (offset < 2 * k && list < list_count) {
            const std::size_t index =
                (static_cast<std::size_t>(row) * list_count + list) * k + offset % k;
            scores[offset] = input_values[index];
            ids[offset] = input_ids[index];
        } else {
            scores[offset] = -CUDART_INF_F;
            ids[offset] = INT_MAX;
        }
    }
    __syncthreads();

    for (int width = 2; width <= sort_length; width *= 2) {
        for (int distance = width / 2; distance > 0; distance /= 2) {
            for (int offset = lane; offset < sort_length; offset += kTileSize) {
                const int partner = offset ^ distance;
                if (offset < partner) {
                    const bool best_first = (offset & width) == 0;
                    const bool partner_first = comes_before(scores[partner], ids[partner],
                                                            scores[offset], ids[offset]);
                    if (partner_first == best_first) {
                        const float score = scores[offset];
                        const int id = ids[offset];
                        scores[offset] = scores[partner];
                        ids[offset] = ids[partner];
                        scores[partner] = score;
                        ids[partner] = id;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (lane < k) {
        const std::size_t index =
            (static_cast<std::size_t>(row) * next_count + pair) * k + lane;
        output_values[index] = scores[lane];
        output_ids[index] = ids[lane];
    }
}

int merge_sort_length(int k) {
    int length = 1;
    while (length < 2 * k)
        length *= 2;
    return length;
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

std::size_t hierarchical_workspace_elements(int batch, int vocab, int k) {
    const int tile_count = local_tile_count(vocab);
    if (batch <= 0 || tile_count == 0 || k <= 0 || k > vocab || k > kTileSize)
        return 0;
    const auto tiles = static_cast<std::size_t>(batch) * tile_count;
    const auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (tiles > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        tiles > max_elements / static_cast<std::size_t>(k))
        return 0;
    return tile_count == 1 ? 0 : tiles * k;
}

cudaError_t hierarchical_topk(const float* input, int batch, int vocab, int k,
                              float* output_values, int* output_ids,
                              HierarchicalWorkspace workspace, cudaStream_t stream) {
    const int tile_count = local_tile_count(vocab);
    if (batch <= 0 || tile_count == 0 || k <= 0 || k > vocab || k > kTileSize ||
        !input || !output_values || !output_ids)
        return cudaErrorInvalidValue;
    if (tile_count == 1)
        return local_tile_topk(input, batch, vocab, k, output_values, output_ids, stream);

    const auto capacity = hierarchical_workspace_elements(batch, vocab, k);
    if (capacity == 0 || !workspace.values_a || !workspace.ids_a ||
        !workspace.values_b || !workspace.ids_b || workspace.capacity < capacity)
        return cudaErrorInvalidValue;

    auto error = local_tile_topk(input, batch, vocab, k,
                                 workspace.values_a, workspace.ids_a, stream);
    if (error != cudaSuccess)
        return error;

    const float* current_values = workspace.values_a;
    const int* current_ids = workspace.ids_a;
    int count = tile_count;
    bool use_b = true;
    const int sort_length = merge_sort_length(k);
    while (count > 1) {
        const int next_count = 1 + (count - 1) / 2;
        const bool final_round = next_count == 1;
        float* next_values = final_round ? output_values
                             : use_b ? workspace.values_b : workspace.values_a;
        int* next_ids = final_round ? output_ids
                        : use_b ? workspace.ids_b : workspace.ids_a;
        const auto blocks = static_cast<int>(static_cast<std::size_t>(batch) * next_count);
        merge_pairs_kernel<<<blocks, kTileSize, 0, stream>>>(
            current_values, current_ids, count, next_count, k, sort_length,
            next_values, next_ids);
        error = cudaGetLastError();
        if (error != cudaSuccess)
            return error;
        current_values = next_values;
        current_ids = next_ids;
        count = next_count;
        use_b = !use_b;
    }
    return cudaSuccess;
}

}  // namespace topk
