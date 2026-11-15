#include "full_sort_sampling.cuh"

#include <cstdlib>
#include <iostream>

namespace {

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
T* allocate(std::size_t count) {
    T* pointer = nullptr;
    check(cudaMalloc(&pointer, count * sizeof(T)));
    return pointer;
}

}  // namespace

int main() {
    constexpr int batch = 2;
    constexpr int vocab = 5;  // Exercise padding and original token IDs.
    constexpr int k = 3;
    const float input[] = {1, -2, 3, -5, 2, -4, 0, 0, -3, 0};
    const float random_values[] = {0.8f, 0.5f};
    // Row 0 selects from IDs [2,4,0]. Row 1 ties and selects from [1,2,4].
    const std::int32_t expected[] = {4, 2};

    const auto capacity = topk::full_sort_workspace_elements(batch, vocab);
    topk::FullSortSamplingBuffers buffers{
        {allocate<float>(capacity), allocate<int>(capacity), capacity},
        allocate<float>(batch * k), allocate<std::int32_t>(batch * k),
        allocate<float>(batch * k), allocate<float>(batch * k),
        allocate<std::int32_t>(batch),
    };
    auto* logits = allocate<float>(batch * vocab);
    auto* randoms = allocate<float>(batch);
    check(cudaMemcpy(logits, input, sizeof(input), cudaMemcpyHostToDevice));
    check(cudaMemcpy(randoms, random_values, sizeof(random_values),
                     cudaMemcpyHostToDevice));

    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(topk::full_sort_sample(logits, randoms, batch, vocab, k,
                                 1.0f, buffers, stream));
    check(cudaStreamSynchronize(stream));

    std::int32_t sampled[batch];
    check(cudaMemcpy(sampled, buffers.sampled_ids, sizeof(sampled),
                     cudaMemcpyDeviceToHost));

    check(cudaStreamDestroy(stream));
    check(cudaFree(logits));
    check(cudaFree(randoms));
    check(cudaFree(buffers.sort.values));
    check(cudaFree(buffers.sort.ids));
    check(cudaFree(buffers.selected_scores));
    check(cudaFree(buffers.selected_ids));
    check(cudaFree(buffers.probabilities));
    check(cudaFree(buffers.cdf));
    check(cudaFree(buffers.sampled_ids));

    for (int row = 0; row < batch; ++row) {
        if (sampled[row] != expected[row]) {
            std::cerr << "Row " << row << ": expected token " << expected[row]
                      << ", got " << sampled[row] << '\n';
            return EXIT_FAILURE;
        }
    }
    std::cout << "PASS: full sort and parallel sampling returned original token IDs\n";
}
