#include <climits>
#include <limits>
#include <math_constants.h>

#include "topk.cuh"

namespace topk {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 65535;
constexpr int kMaxVocabularySize = 1 << 30;

// Host helpers: choose the padded row length and the launch size.
std::size_t next_power_of_two(int vocab_size) {
    if (vocab_size <= 0 || vocab_size > kMaxVocabularySize) {
        return 0;
    }

    std::size_t padded_size = 1;
    while (padded_size < static_cast<std::size_t>(vocab_size)) {
        padded_size *= 2;
    }
    return padded_size;
}

int blocks_needed(std::size_t element_count) {
    const auto block_count = (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return static_cast<int>(block_count > kMaxBlocks ? kMaxBlocks : block_count);
}

// Our final ordering: larger scores first, smaller original IDs break ties.
__device__ bool comes_before(float score_a, int token_a, float score_b, int token_b) {
    if (score_a != score_b) {
        return score_a > score_b;
    }
    return token_a < token_b;
}

// STEP 1: Copy each row into workspace and fill unused columns with padding.
__global__ void copy_and_pad_kernel(const float* input_scores, float* work_scores,
                                    int* work_token_ids, std::size_t total_elements,
                                    std::size_t padded_vocab_size, int vocab_size) {
    const std::size_t first_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t grid_stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t index = first_index; index < total_elements;
         index += grid_stride) {
        const auto row = index / padded_vocab_size;
        const auto column = index % padded_vocab_size;

        if (column < static_cast<std::size_t>(vocab_size)) {
            work_scores[index] = input_scores[row * vocab_size + column];
            work_token_ids[index] = static_cast<int>(column);
        } else {
            // Finite input scores always rank ahead of these padded entries.
            work_scores[index] = -CUDART_INF_F;
            work_token_ids[index] = INT_MAX;
        }
    }
}

// STEP 2: Perform ONE comparison stage of the bitonic sorting network.
// Each pair has one writer. The next stage is a separate launch on the same
// stream, so it cannot start until this stage has finished across all blocks.
__global__ void compare_pairs_kernel(float* work_scores, int* work_token_ids,
                                     std::size_t total_elements,
                                     std::size_t padded_vocab_size,
                                     std::size_t group_size,
                                     std::size_t partner_distance) {
    const std::size_t first_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t grid_stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t index = first_index; index < total_elements;
         index += grid_stride) {
        const auto column = index % padded_vocab_size;
        const auto partner_column = column ^ partner_distance;

        // The lower-index thread handles the pair; the other skips it.
        if (partner_column <= column) {
            continue;
        }

        const auto row_start = index - column;
        const auto partner_index = row_start + partner_column;
        const float left_score = work_scores[index];
        const float right_score = work_scores[partner_index];
        const int left_token = work_token_ids[index];
        const int right_token = work_token_ids[partner_index];

        // Intermediate groups alternate directions. The last group covers
        // the entire row and puts the best scores first everywhere.
        const bool best_first = (column & group_size) == 0;
        bool should_swap;
        if (best_first) {
            should_swap =
                comes_before(right_score, right_token, left_score, left_token);
        } else {
            should_swap =
                comes_before(left_score, left_token, right_score, right_token);
        }

        if (should_swap) {
            work_scores[index] = right_score;
            work_token_ids[index] = right_token;
            work_scores[partner_index] = left_score;
            work_token_ids[partner_index] = left_token;
        }
    }
}

// STEP 3: Each row is now fully sorted. Copy its first k entries to output.
__global__ void copy_top_k_kernel(const float* work_scores, const int* work_token_ids,
                                  float* output_scores, int* output_token_ids,
                                  std::size_t output_count,
                                  std::size_t padded_vocab_size, int k) {
    const std::size_t first_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t grid_stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t index = first_index; index < output_count; index += grid_stride) {
        const auto row = index / k;
        const auto rank = index % k;
        const auto source_index = row * padded_vocab_size + rank;

        output_scores[index] = work_scores[source_index];
        output_token_ids[index] = work_token_ids[source_index];
    }
}

}  // namespace

std::size_t full_sort_workspace_elements(int batch_size, int vocab_size) {
    const auto padded_vocab_size = next_power_of_two(vocab_size);
    if (batch_size <= 0 || padded_vocab_size == 0) {
        return 0;
    }

    // Make sure either workspace allocation fits in a size_t byte count.
    const auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (static_cast<std::size_t>(batch_size) > max_elements / padded_vocab_size) {
        return 0;
    }
    return static_cast<std::size_t>(batch_size) * padded_vocab_size;
}

// Read this function first: it launches the three steps defined above.
cudaError_t full_sort(const float* input_scores, int batch_size, int vocab_size, int k,
                      float* output_scores, int* output_token_ids,
                      FullSortWorkspace workspace, cudaStream_t stream) {
    const auto total_elements = full_sort_workspace_elements(batch_size, vocab_size);
    const bool invalid_dimensions = total_elements == 0 || k <= 0 || k > vocab_size;
    const bool missing_buffer = !input_scores || !output_scores || !output_token_ids ||
                                !workspace.values || !workspace.ids;
    if (invalid_dimensions || missing_buffer || workspace.capacity < total_elements) {
        return cudaErrorInvalidValue;
    }

    const auto padded_vocab_size = next_power_of_two(vocab_size);
    const int block_count = blocks_needed(total_elements);

    // 1. Prepare score/ID pairs in padded workspace.
    copy_and_pad_kernel<<<block_count, kThreadsPerBlock, 0, stream>>>(
        input_scores, workspace.values, workspace.ids, total_elements,
        padded_vocab_size, vocab_size);

    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }

    // 2. Sort groups of 2, then 4, then 8, ... up to the entire padded row.
    for (std::size_t group_size = 2; group_size <= padded_vocab_size; group_size *= 2) {
        for (std::size_t partner_distance = group_size / 2; partner_distance > 0;
             partner_distance /= 2) {
            compare_pairs_kernel<<<block_count, kThreadsPerBlock, 0, stream>>>(
                workspace.values, workspace.ids, total_elements, padded_vocab_size,
                group_size, partner_distance);

            error = cudaGetLastError();
            if (error != cudaSuccess) {
                return error;
            }
        }
    }

    // 3. Return the first k sorted score/ID pairs from every row.
    const auto output_count = static_cast<std::size_t>(batch_size) * k;
    copy_top_k_kernel<<<blocks_needed(output_count), kThreadsPerBlock, 0, stream>>>(
        workspace.values, workspace.ids, output_scores, output_token_ids, output_count,
        padded_vocab_size, k);

    return cudaGetLastError();
}

}  // namespace topk
