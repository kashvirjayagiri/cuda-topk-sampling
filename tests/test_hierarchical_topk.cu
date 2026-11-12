#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "topk.cuh"

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

#define CHECK(operation) check_cuda((operation), #operation)

void test(int batch, int vocab, int k, int distribution) {
    std::mt19937 rng(451);
    std::normal_distribution<float> normal;
    std::vector<float> input(static_cast<std::size_t>(batch) * vocab);
    for (int row = 0; row < batch; ++row) {
        for (int id = 0; id < vocab; ++id) {
            input[static_cast<std::size_t>(row) * vocab + id] =
                distribution == 0 ? normal(rng)
                : distribution == 1 ? static_cast<float>(rng() % 5) - 2.0f
                : distribution == 2 ? -3.0f
                : distribution == 3 ? static_cast<float>(1000 - 2 * (id % 256) - (id / 256))
                                    : static_cast<float>((id == 255 || id == 256) ? 1000 : -1000 - id);
        }
    }

    float *device_input, *device_scores;
    int* device_ids;
    CHECK(cudaMalloc(&device_input, input.size() * sizeof(float)));
    CHECK(cudaMalloc(&device_scores, static_cast<std::size_t>(batch) * k * sizeof(float)));
    CHECK(cudaMalloc(&device_ids, static_cast<std::size_t>(batch) * k * sizeof(int)));
    CHECK(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float),
                     cudaMemcpyHostToDevice));

    const auto capacity = topk::hierarchical_workspace_elements(batch, vocab, k);
    topk::HierarchicalWorkspace workspace{nullptr, nullptr, nullptr, nullptr, capacity};
    if (capacity) {
        CHECK(cudaMalloc(&workspace.values_a, capacity * sizeof(float)));
        CHECK(cudaMalloc(&workspace.ids_a, capacity * sizeof(int)));
        CHECK(cudaMalloc(&workspace.values_b, capacity * sizeof(float)));
        CHECK(cudaMalloc(&workspace.ids_b, capacity * sizeof(int)));
    }
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    CHECK(topk::hierarchical_topk(device_input, batch, vocab, k, device_scores,
                                  device_ids, workspace, stream));
    CHECK(cudaStreamSynchronize(stream));

    std::vector<float> scores(static_cast<std::size_t>(batch) * k), unchanged(input.size());
    std::vector<int> ids(static_cast<std::size_t>(batch) * k);
    CHECK(cudaMemcpy(scores.data(), device_scores, scores.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(ids.data(), device_ids, ids.size() * sizeof(int),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(unchanged.data(), device_input, input.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    if (input != unchanged) {
        std::cerr << "Hierarchical selector modified input\n";
        std::exit(1);
    }

    for (int row = 0; row < batch; ++row) {
        std::vector<int> expected(vocab);
        std::iota(expected.begin(), expected.end(), 0);
        std::stable_sort(expected.begin(), expected.end(), [&](int a, int b) {
            return input[static_cast<std::size_t>(row) * vocab + a] >
                   input[static_cast<std::size_t>(row) * vocab + b];
        });
        for (int rank = 0; rank < k; ++rank) {
            const auto index = static_cast<std::size_t>(row) * k + rank;
            const int id = expected[rank];
            if (ids[index] != id || ids[index] == INT_MAX ||
                scores[index] != input[static_cast<std::size_t>(row) * vocab + id]) {
                std::cerr << "Mismatch B=" << batch << " V=" << vocab << " k=" << k
                          << " distribution=" << distribution << " row=" << row
                          << " rank=" << rank << '\n';
                std::exit(1);
            }
        }
    }

    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(device_input));
    CHECK(cudaFree(device_scores));
    CHECK(cudaFree(device_ids));
    if (capacity) {
        CHECK(cudaFree(workspace.values_a));
        CHECK(cudaFree(workspace.ids_a));
        CHECK(cudaFree(workspace.values_b));
        CHECK(cudaFree(workspace.ids_b));
    }
}

int main() {
    float placeholder = 0.0f;
    int placeholder_id = 0;
    topk::HierarchicalWorkspace empty{nullptr, nullptr, nullptr, nullptr, 0};
    if (topk::hierarchical_workspace_elements(1, 256, 100) != 0 ||
        topk::hierarchical_workspace_elements(8, 257, 10) != 160 ||
        topk::hierarchical_workspace_elements(0, 257, 10) != 0 ||
        topk::hierarchical_workspace_elements(1, 257, 257) != 0 ||
        topk::hierarchical_workspace_elements(INT_MAX, INT_MAX, 256) != 0 ||
        topk::hierarchical_topk(nullptr, 1, 257, 10, &placeholder,
                                &placeholder_id, empty) != cudaErrorInvalidValue ||
        topk::hierarchical_topk(&placeholder, 0, 257, 10, &placeholder,
                                &placeholder_id, empty) != cudaErrorInvalidValue ||
        topk::hierarchical_topk(&placeholder, 1, 0, 10, &placeholder,
                                &placeholder_id, empty) != cudaErrorInvalidValue ||
        topk::hierarchical_topk(&placeholder, 1, 257, 0, &placeholder,
                                &placeholder_id, empty) != cudaErrorInvalidValue ||
        topk::hierarchical_topk(&placeholder, 1, 257, 10, &placeholder,
                                &placeholder_id, empty) != cudaErrorInvalidValue ||
        topk::hierarchical_topk(&placeholder, 1, 257, 257, &placeholder,
                                &placeholder_id, empty) != cudaErrorInvalidValue)
        return 1;
    topk::HierarchicalWorkspace short_workspace{&placeholder, &placeholder_id,
                                                 &placeholder, &placeholder_id, 19};
    if (topk::hierarchical_topk(&placeholder, 1, 257, 10, &placeholder,
                                &placeholder_id, short_workspace) != cudaErrorInvalidValue)
        return 1;

    for (int batch : {1, 8, 32}) {
        for (int vocab : {1, 17, 250, 256, 257, 259, 512, 513, 1025}) {
            for (int k : {1, 10, 50, 100, 256}) {
                if (k <= vocab) {
                    for (int distribution = 0; distribution < 3; ++distribution)
                        test(batch, vocab, k, distribution);
                }
            }
        }
    }
    test(1, 512, 100, 3);  // Winners alternate between adjacent tiles.
    test(1, 512, 1, 4);    // Equal winners are IDs 255 and 256.
    test(8, 512, 100, 4);
    for (int vocab : {8192, 32768, 50000, 131072})
        test(1, vocab, 100, 0);
    test(32, 131072, 100, 1);
    std::cout << "PASS: global Top-k, T=1/2/3/5, odd merges, ties, padding, "
                 "large vocabularies, input preservation\n";
}
