#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <limits>
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
    for (auto& score : input) {
        score = distribution == 0 ? normal(rng)
                : distribution == 1 ? static_cast<float>(rng() % 5) - 2.0f
                                    : -3.0f;
    }

    const int tile_count = topk::local_tile_count(vocab);
    const std::size_t output_count = static_cast<std::size_t>(batch) * tile_count * k;
    float *device_input, *device_scores;
    int* device_ids;
    CHECK(cudaMalloc(&device_input, input.size() * sizeof(float)));
    CHECK(cudaMalloc(&device_scores, output_count * sizeof(float)));
    CHECK(cudaMalloc(&device_ids, output_count * sizeof(int)));
    CHECK(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    CHECK(topk::local_tile_topk(device_input, batch, vocab, k, device_scores,
                                device_ids, stream));
    CHECK(cudaStreamSynchronize(stream));

    std::vector<float> scores(output_count), unchanged(input.size());
    std::vector<int> ids(output_count);
    CHECK(cudaMemcpy(scores.data(), device_scores, output_count * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(ids.data(), device_ids, output_count * sizeof(int),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(unchanged.data(), device_input, input.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    if (input != unchanged) {
        std::cerr << "Local tile stage modified input\n";
        std::exit(1);
    }

    for (int row = 0; row < batch; ++row) {
        for (int tile = 0; tile < tile_count; ++tile) {
            std::vector<int> expected(256);
            for (int lane = 0; lane < 256; ++lane)
                expected[lane] = tile * 256 + lane < vocab ? tile * 256 + lane : INT_MAX;
            std::stable_sort(expected.begin(), expected.end(), [&](int a, int b) {
                const float av = a == INT_MAX ? -std::numeric_limits<float>::infinity()
                                              : input[static_cast<std::size_t>(row) * vocab + a];
                const float bv = b == INT_MAX ? -std::numeric_limits<float>::infinity()
                                              : input[static_cast<std::size_t>(row) * vocab + b];
                return av > bv;
            });
            for (int rank = 0; rank < k; ++rank) {
                const auto index = (static_cast<std::size_t>(row) * tile_count + tile) * k + rank;
                const int id = expected[rank];
                const float score = id == INT_MAX ? -std::numeric_limits<float>::infinity()
                                                  : input[static_cast<std::size_t>(row) * vocab + id];
                if (ids[index] != id || scores[index] != score) {
                    std::cerr << "Mismatch B=" << batch << " V=" << vocab << " k=" << k
                              << " distribution=" << distribution << " row=" << row
                              << " tile=" << tile << " rank=" << rank << '\n';
                    std::exit(1);
                }
            }
        }
    }

    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(device_input));
    CHECK(cudaFree(device_scores));
    CHECK(cudaFree(device_ids));
}

int main() {
    float placeholder = 0.0f;
    int placeholder_id = 0;
    if (topk::local_tile_count(0) != 0 || topk::local_tile_count(1) != 1 ||
        topk::local_tile_count(256) != 1 || topk::local_tile_count(257) != 2 ||
        topk::local_tile_topk(nullptr, 1, 256, 10, nullptr, nullptr) != cudaErrorInvalidValue ||
        topk::local_tile_topk(&placeholder, 0, 256, 10, &placeholder, &placeholder_id) != cudaErrorInvalidValue ||
        topk::local_tile_topk(&placeholder, 1, 0, 1, &placeholder, &placeholder_id) != cudaErrorInvalidValue ||
        topk::local_tile_topk(&placeholder, 1, 256, 0, &placeholder, &placeholder_id) != cudaErrorInvalidValue ||
        topk::local_tile_topk(&placeholder, 1, 10, 11, &placeholder, &placeholder_id) != cudaErrorInvalidValue ||
        topk::local_tile_topk(&placeholder, 1, 257, 257, &placeholder, &placeholder_id) != cudaErrorInvalidValue)
        return 1;

    for (int distribution = 0; distribution < 3; ++distribution) {
        for (int batch : {1, 8, 32}) {
            for (int vocab : {1, 17, 250, 256, 257, 259, 512, 513, 50257}) {
                for (int k : {1, 10, 50, 100}) {
                    if (k <= vocab)
                        test(batch, vocab, k, distribution);
                }
            }
        }
    }
    std::cout << "PASS: local tiles, partial tiles, ties, negative/equal scores, "
                 "batches, padding, input preservation\n";
}
