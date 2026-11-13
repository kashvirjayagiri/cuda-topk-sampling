#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "rejection_sampler.cuh"

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}
#define CHECK(operation) check_cuda((operation), #operation)

void test(int batch, int vocab, int k, int kind, float temperature, int draws) {
    std::mt19937 rng(451);
    std::normal_distribution<float> normal;
    std::vector<float> input(static_cast<std::size_t>(batch) * vocab);
    for (int row = 0; row < batch; ++row) {
        for (int id = 0; id < vocab; ++id) {
            float score = kind == 0 ? normal(rng)
                        : kind == 1 ? static_cast<float>(rng() % 5) - 2.0f
                        : kind == 2 ? -3.0f
                        : kind == 3 ? ((id == 255 || id == 256) ? 100.0f : -1000.0f - id)
                                    : (id % 3 == 0 ? -1.0e30f : id % 3 == 1 ? -0.0f : 1.0e30f);
            input[static_cast<std::size_t>(row) * vocab + id] = score;
        }
    }
    std::vector<std::vector<int>> ranks(batch, std::vector<int>(vocab));
    for (int row = 0; row < batch; ++row) {
        std::vector<int> ordered(vocab);
        std::iota(ordered.begin(), ordered.end(), 0);
        std::stable_sort(ordered.begin(), ordered.end(), [&](int a, int b) {
            return input[static_cast<std::size_t>(row) * vocab + a] >
                   input[static_cast<std::size_t>(row) * vocab + b];
        });
        for (int rank = 0; rank < vocab; ++rank)
            ranks[row][ordered[rank]] = rank;
    }

    float* device_input;
    int *device_ids, *device_attempts;
    CHECK(cudaMalloc(&device_input, input.size() * sizeof(float)));
    CHECK(cudaMalloc(&device_ids, batch * sizeof(int)));
    CHECK(cudaMalloc(&device_attempts, batch * sizeof(int)));
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(cudaMemcpyAsync(device_input, input.data(), input.size() * sizeof(float),
                          cudaMemcpyHostToDevice, stream));

    for (int draw = 0; draw < draws; ++draw) {
        std::vector<int> first_ids(batch), first_attempts(batch);
        for (int replay = 0; replay < 2; ++replay) {
            CHECK(topk::rejection_sample_topk(device_input, batch, vocab, k,
                                              temperature, 451, draw,
                                              device_ids, device_attempts, stream));
            CHECK(cudaStreamSynchronize(stream));
            std::vector<int> ids(batch), attempts(batch);
            CHECK(cudaMemcpy(ids.data(), device_ids, batch * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(attempts.data(), device_attempts, batch * sizeof(int),
                             cudaMemcpyDeviceToHost));
            for (int row = 0; row < batch; ++row) {
                if (ids[row] < 0 || ids[row] >= vocab || ranks[row][ids[row]] >= k ||
                    attempts[row] < 1 || attempts[row] > vocab - k + 1 ||
                    ((k == 1 || k == vocab) && attempts[row] != 1)) {
                    std::cerr << "Invalid rejection sample B=" << batch << " V=" << vocab
                              << " k=" << k << " kind=" << kind << " T=" << temperature
                              << " draw=" << draw << " row=" << row << '\n';
                    std::exit(1);
                }
            }
            if (replay == 0) {
                first_ids = ids;
                first_attempts = attempts;
            } else if (ids != first_ids || attempts != first_attempts) {
                std::cerr << "Same seed/draw did not replay deterministically\n";
                std::exit(1);
            }
        }
    }
    std::vector<float> unchanged(input.size());
    CHECK(cudaMemcpy(unchanged.data(), device_input, input.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    if (std::memcmp(unchanged.data(), input.data(), input.size() * sizeof(float)) != 0) {
        std::cerr << "Rejection sampler modified logits\n";
        std::exit(1);
    }
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(device_input));
    CHECK(cudaFree(device_ids));
    CHECK(cudaFree(device_attempts));
}

int main() {
    float score = 0.0f;
    int output = 0;
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    if (topk::rejection_sample_topk(nullptr, 1, 3, 2, 1, 0, 0, &output, &output) !=
            cudaErrorInvalidValue ||
        topk::rejection_sample_topk(&score, 0, 3, 2, 1, 0, 0, &output, &output) !=
            cudaErrorInvalidValue ||
        topk::rejection_sample_topk(&score, 1, 0, 1, 1, 0, 0, &output, &output) !=
            cudaErrorInvalidValue ||
        topk::rejection_sample_topk(&score, 1, 3, 4, 1, 0, 0, &output, &output) !=
            cudaErrorInvalidValue ||
        topk::rejection_sample_topk(&score, 1, 3, 0, 1, 0, 0, &output, &output) !=
            cudaErrorInvalidValue ||
        topk::rejection_sample_topk(&score, 1, 3, 2, 0, 0, 0, &output, &output) !=
            cudaErrorInvalidValue ||
        topk::rejection_sample_topk(&score, 1, 3, 2, -1, 0, 0, &output, &output) !=
            cudaErrorInvalidValue ||
        topk::rejection_sample_topk(&score, 1, 3, 2, inf, 0, 0, &output, &output) !=
            cudaErrorInvalidValue ||
        topk::rejection_sample_topk(&score, 1, 3, 2, nan, 0, 0, &output, &output) !=
            cudaErrorInvalidValue ||
        topk::rejection_sample_topk(&score, 1, 3, 2, 1, 0, 0, nullptr, &output) !=
            cudaErrorInvalidValue)
        return 1;

    for (int batch : {1, 8, 32}) {
        for (int vocab : {1, 3, 17, 257, 513}) {
            for (int k : {1, 2, 10, 50, 100, 300}) {
                if (k <= vocab) {
                    for (int kind : {0, 1, 2})
                        test(batch, vocab, k, kind, 1.0f, 4);
                }
            }
        }
    }
    test(1, 513, 513, 1, 0.7f, 4);
    test(8, 513, 10, 3, 2.5f, 8);
    test(8, 513, 50, 4, 0.1f, 8);
    for (int vocab : {8192, 32768, 50000, 131072})
        test(1, vocab, 100, 0, 1.0f, 2);
    test(32, 131072, 100, 1, 1.0f, 2);
    std::cout << "PASS: membership, ties, temperatures, replay, attempts bound, "
                 "input preservation, nondefault stream, large V\n";
}
