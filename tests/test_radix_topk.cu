#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
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
    const float special[] = {-1.0e30f, -1.0f, -std::nextafter(1.0f, 2.0f),
                             -1.0f, -0.0f, 0.0f, 1.0f,
                             std::nextafter(1.0f, 2.0f), 1.0e30f};
    std::vector<float> input(static_cast<std::size_t>(batch) * vocab);
    for (int row = 0; row < batch; ++row) {
        for (int id = 0; id < vocab; ++id) {
            float score;
            if (distribution == 0)
                score = normal(rng);
            else if (distribution == 1)
                score = static_cast<float>(rng() % 9) - 4.0f;
            else if (distribution == 2)
                score = -3.0f;
            else if (distribution == 3)
                score = special[id % 9];
            else
                score = (id == 255 || id == 256) ? 100.0f : -1000.0f - id;
            input[static_cast<std::size_t>(row) * vocab + id] = score;
        }
    }

    std::size_t workspace_bytes = 0;
    CHECK(topk::radix_topk_workspace_bytes(batch, vocab, k, &workspace_bytes));
    if (workspace_bytes == 0) {
        std::cerr << "Empty radix workspace\n";
        std::exit(1);
    }
    float *device_input, *device_scores;
    int* device_ids;
    void* workspace;
    CHECK(cudaMalloc(&device_input, input.size() * sizeof(float)));
    CHECK(cudaMalloc(&device_scores, static_cast<std::size_t>(batch) * k * sizeof(float)));
    CHECK(cudaMalloc(&device_ids, static_cast<std::size_t>(batch) * k * sizeof(int)));
    CHECK(cudaMalloc(&workspace, workspace_bytes));
    if (topk::radix_topk(device_input, batch, vocab, k, device_scores, device_ids,
                          {workspace, workspace_bytes - 1}) != cudaErrorInvalidValue) {
        std::cerr << "Accepted undersized radix workspace\n";
        std::exit(1);
    }
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(cudaMemcpyAsync(device_input, input.data(), input.size() * sizeof(float),
                          cudaMemcpyHostToDevice, stream));

    for (int repeat = 0; repeat < 2; ++repeat) {
        CHECK(topk::radix_topk(device_input, batch, vocab, k, device_scores, device_ids,
                               {workspace, workspace_bytes}, stream));
        CHECK(cudaStreamSynchronize(stream));
        std::vector<float> scores(static_cast<std::size_t>(batch) * k), unchanged(input.size());
        std::vector<int> ids(static_cast<std::size_t>(batch) * k);
        CHECK(cudaMemcpy(scores.data(), device_scores, scores.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(ids.data(), device_ids, ids.size() * sizeof(int),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(unchanged.data(), device_input, input.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));
        if (std::memcmp(input.data(), unchanged.data(), input.size() * sizeof(float)) != 0) {
            std::cerr << "Radix selector modified input\n";
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
                const float expected_score = input[static_cast<std::size_t>(row) * vocab + expected[rank]];
                if (ids[index] != expected[rank] || ids[index] == INT_MAX ||
                    std::memcmp(&scores[index], &expected_score, sizeof(float)) != 0) {
                    std::cerr << "Mismatch B=" << batch << " V=" << vocab << " k=" << k
                              << " distribution=" << distribution << " repeat=" << repeat
                              << " row=" << row << " rank=" << rank << '\n';
                    std::exit(1);
                }
            }
        }
    }
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(device_input));
    CHECK(cudaFree(device_scores));
    CHECK(cudaFree(device_ids));
    CHECK(cudaFree(workspace));
}

int main() {
    std::size_t bytes = 99;
    if (topk::radix_topk_workspace_bytes(0, 17, 10, &bytes) != cudaErrorInvalidValue ||
        bytes != 0 ||
        topk::radix_topk_workspace_bytes(1, 17, 18, &bytes) != cudaErrorInvalidValue ||
        topk::radix_topk_workspace_bytes(INT_MAX, 2, 1, &bytes) != cudaErrorInvalidValue ||
        topk::radix_topk_workspace_bytes(1, 17, 10, nullptr) != cudaErrorInvalidValue ||
        topk::radix_topk(nullptr, 1, 17, 10, nullptr, nullptr, {nullptr, 0}) !=
            cudaErrorInvalidValue)
        return 1;

    for (int batch : {1, 8, 32}) {
        for (int vocab : {1, 2, 7, 17, 255, 256, 257, 513, 1025}) {
            for (int k : {1, 10, 50, 100, 257}) {
                if (k <= vocab) {
                    for (int distribution = 0; distribution < 4; ++distribution)
                        test(batch, vocab, k, distribution);
                }
            }
        }
    }
    for (int vocab : {2, 17, 257, 513})
        test(1, vocab, vocab, 3);
    test(8, 513, 513, 1);
    test(8, 513, 50, 4);
    for (int vocab : {8192, 32768, 50000, 131072})
        test(1, vocab, 100, 0);
    test(32, 131072, 100, 1);
    std::size_t large_workspace = 0;
    CHECK(topk::radix_topk_workspace_bytes(32, 131072, 100, &large_workspace));
    if (large_workspace < 68ULL * 1024 * 1024 + 51200 + 516)
        return 1;
    std::cout << "PASS: radix global Top-k, signed zero, close/wide scores, ties, "
                 "k > 256, repeated calls, nondefault stream, large vocabularies\n"
              << "Workspace B=32 V=131072 k=100: " << large_workspace << " bytes\n";
}
