#include "hierarchical_topk_sampling.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(operation) check((operation), #operation)

template <typename T>
T* allocate(std::size_t count) {
    T* pointer = nullptr;
    CHECK(cudaMalloc(&pointer, count * sizeof(T)));
    return pointer;
}

void test(const std::vector<float>& input, int batch, int vocab, int k,
          float temperature, const std::vector<float>& randoms) {
    const auto capacity = topk::hierarchical_workspace_elements(batch, vocab, k);
    topk::HierarchicalWorkspace workspace{nullptr, nullptr, nullptr, nullptr, capacity};
    if (capacity) {
        workspace.values_a = allocate<float>(capacity);
        workspace.ids_a = allocate<int>(capacity);
        workspace.values_b = allocate<float>(capacity);
        workspace.ids_b = allocate<int>(capacity);
    }
    topk::HierarchicalTopKSamplingBuffers buffers{
        workspace, allocate<float>(static_cast<std::size_t>(batch) * k),
        allocate<std::int32_t>(static_cast<std::size_t>(batch) * k),
        allocate<float>(static_cast<std::size_t>(batch) * k),
        allocate<float>(static_cast<std::size_t>(batch) * k),
        allocate<std::int32_t>(batch),
    };
    auto* device_input = allocate<float>(input.size());
    auto* device_randoms = allocate<float>(randoms.size());
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(cudaMemcpyAsync(device_input, input.data(), input.size() * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(device_randoms, randoms.data(), randoms.size() * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    CHECK(topk::hierarchical_topk_sample(device_input, device_randoms, batch, vocab, k,
                                          temperature, buffers, stream));
    CHECK(cudaStreamSynchronize(stream));

    std::vector<float> selected_scores(static_cast<std::size_t>(batch) * k);
    std::vector<std::int32_t> selected_ids(static_cast<std::size_t>(batch) * k);
    std::vector<std::int32_t> sampled_ids(batch);
    std::vector<float> unchanged(input.size());
    CHECK(cudaMemcpy(selected_scores.data(), buffers.selected_scores,
                     selected_scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(selected_ids.data(), buffers.selected_ids,
                     selected_ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sampled_ids.data(), buffers.sampled_ids,
                     sampled_ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(unchanged.data(), device_input,
                     input.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (std::memcmp(unchanged.data(), input.data(), input.size() * sizeof(float)) != 0) {
        std::cerr << "Input logits changed\n";
        std::exit(EXIT_FAILURE);
    }

    for (int row = 0; row < batch; ++row) {
        std::vector<int> ordered(vocab);
        std::iota(ordered.begin(), ordered.end(), 0);
        std::stable_sort(ordered.begin(), ordered.end(), [&](int a, int b) {
            return input[static_cast<std::size_t>(row) * vocab + a] >
                   input[static_cast<std::size_t>(row) * vocab + b];
        });
        double max_score = static_cast<double>(input[static_cast<std::size_t>(row) * vocab + ordered[0]]);
        std::vector<double> weights(k);
        double total = 0.0;
        for (int rank = 0; rank < k; ++rank) {
            const auto index = static_cast<std::size_t>(row) * k + rank;
            const float expected_score = input[static_cast<std::size_t>(row) * vocab + ordered[rank]];
            if (selected_ids[index] != ordered[rank] || selected_scores[index] != expected_score) {
                std::cerr << "Selected Top-k mismatch B=" << batch << " V=" << vocab
                          << " k=" << k << " row=" << row << " rank=" << rank << '\n';
                std::exit(EXIT_FAILURE);
            }
            weights[rank] = std::exp((static_cast<double>(expected_score) - max_score) /
                                     static_cast<double>(temperature));
            total += weights[rank];
        }
        double cumulative = 0.0;
        int expected_sample = ordered[k - 1];
        for (int rank = 0; rank < k; ++rank) {
            cumulative = std::min(1.0, cumulative + weights[rank] / total);
            if (rank == k - 1)
                cumulative = 1.0;
            if (cumulative > static_cast<double>(randoms[row])) {
                expected_sample = ordered[rank];
                break;
            }
        }
        if (sampled_ids[row] != expected_sample) {
            std::cerr << "Sample mismatch B=" << batch << " V=" << vocab
                      << " k=" << k << " row=" << row << " expected=" << expected_sample
                      << " actual=" << sampled_ids[row] << '\n';
            std::exit(EXIT_FAILURE);
        }
    }

    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(device_input));
    CHECK(cudaFree(device_randoms));
    if (capacity) {
        CHECK(cudaFree(workspace.values_a));
        CHECK(cudaFree(workspace.ids_a));
        CHECK(cudaFree(workspace.values_b));
        CHECK(cudaFree(workspace.ids_b));
    }
    CHECK(cudaFree(buffers.selected_scores));
    CHECK(cudaFree(buffers.selected_ids));
    CHECK(cudaFree(buffers.probabilities));
    CHECK(cudaFree(buffers.cdf));
    CHECK(cudaFree(buffers.sampled_ids));
}

std::vector<float> scores(int batch, int vocab, int kind) {
    std::mt19937 rng(451);
    std::normal_distribution<float> normal;
    std::vector<float> values(static_cast<std::size_t>(batch) * vocab);
    for (int row = 0; row < batch; ++row) {
        for (int id = 0; id < vocab; ++id) {
            values[static_cast<std::size_t>(row) * vocab + id] =
                kind == 0 ? normal(rng)
                : kind == 1 ? static_cast<float>(rng() % 5) - 2.0f
                : kind == 2 ? -3.0f
                            : static_cast<float>((id == 255 || id == 256) ? 1000 : -1000 - id);
        }
    }
    return values;
}

std::vector<float> randoms(int batch) {
    std::vector<float> values(batch);
    for (int row = 0; row < batch; ++row)
        values[row] = static_cast<float>(((row * 37 + 13) % 97) + 0.5) / 98.0f;
    return values;
}

}  // namespace

int main() {
    // The full-sort integration test's example, now checking selection too.
    test({1, -2, 3, -5, 2, -4, 0, 0, -3, 0}, 2, 5, 3, 1.0f, {0.8f, 0.5f});
    test(scores(1, 257, 0), 1, 257, 50, 1.0f, randoms(1));  // Partial last tile.
    test(scores(8, 513, 1), 8, 513, 10, 0.7f, randoms(8));  // Three tile lists.
    test(scores(8, 1025, 0), 8, 1025, 100, 2.0f, randoms(8));  // Five tile lists.
    test(scores(1, 512, 3), 1, 512, 1, 1.0f, randoms(1));  // IDs 255/256 tie.
    test(scores(8, 512, 3), 8, 512, 10, 1.0f, randoms(8));
    test(scores(32, 259, 2), 32, 259, 50, 1.0f, randoms(32));
    test(scores(32, 8192, 0), 32, 8192, 100, 1.0f, randoms(32));
    std::cout << "PASS: hierarchical selection and parallel sampling, selected IDs, "
                 "FP64 sample reference, ties, partial/odd tiles, input, stream\n";
}
