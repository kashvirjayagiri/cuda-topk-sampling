#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "topk.cuh"

// Stop immediately and show the failing CUDA operation.
void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

#define CHECK(operation) check_cuda((operation), #operation)

void test(int batch, int vocab, int k, int distribution) {
    // 1. Generate a reproducible CPU input.
    std::mt19937 rng(451);
    std::normal_distribution<float> normal;
    std::vector<float> input(batch * vocab);
    for (auto& score : input) {
        if (distribution == 0) {
            score = normal(rng);
        } else if (distribution == 1) {
            score = static_cast<float>(rng() % 5) - 2;
        } else {
            score = -3.0f;
        }
    }
    // 2. Allocate input, output, and temporary buffers on the GPU.
    float *device_input, *out_values, *work_values;
    int *out_ids, *work_ids;
    const auto capacity = topk::full_sort_workspace_elements(batch, vocab);
    CHECK(cudaMalloc(&device_input, input.size() * sizeof(float)));
    CHECK(cudaMalloc(&out_values, batch * k * sizeof(float)));
    CHECK(cudaMalloc(&out_ids, batch * k * sizeof(int)));
    CHECK(cudaMalloc(&work_values, capacity * sizeof(float)));
    CHECK(cudaMalloc(&work_ids, capacity * sizeof(int)));
    CHECK(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    // 3. Run the sorter and wait for GPU completion.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    CHECK(topk::full_sort(device_input, batch, vocab, k, out_values, out_ids,
                          {work_values, work_ids, capacity}, stream));
    CHECK(cudaStreamSynchronize(stream));
    // 4. Read the results back for verification (not performance timing).
    std::vector<float> values(batch * k), unchanged(input.size());
    std::vector<int> ids(batch * k);
    CHECK(cudaMemcpy(values.data(), out_values, values.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(ids.data(), out_ids, ids.size() * sizeof(int),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(unchanged.data(), device_input, unchanged.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    if (input != unchanged) {
        std::cerr << "Input modified\n";
        std::exit(1);
    }
    // 5. Compare against an independent CPU sort, including tie-breaking.
    for (int row = 0; row < batch; ++row) {
        std::vector<int> expected(vocab);
        std::iota(expected.begin(), expected.end(), 0);
        std::sort(expected.begin(), expected.end(), [&](int a, int b) {
            const float av = input[row * vocab + a], bv = input[row * vocab + b];
            return av > bv || (av == bv && a < b);
        });
        for (int j = 0; j < k; ++j) {
            if (ids[row * k + j] != expected[j] ||
                values[row * k + j] != input[row * vocab + expected[j]]) {
                std::cerr << "Mismatch B=" << batch << " V=" << vocab << " k=" << k
                          << " row=" << row << " col=" << j << '\n';
                std::exit(1);
            }
        }
    }
    // 6. Release all resources for this test case.
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(device_input));
    CHECK(cudaFree(out_values));
    CHECK(cudaFree(out_ids));
    CHECK(cudaFree(work_values));
    CHECK(cudaFree(work_ids));
}
int main() {
    if (topk::full_sort(nullptr, 1, 8, 3, nullptr, nullptr, {nullptr, nullptr, 0}) !=
            cudaErrorInvalidValue ||
        topk::full_sort_workspace_elements(0, 8) != 0)
        return 1;
    for (int distribution = 0; distribution < 3; ++distribution) {
        for (int vocab : {1, 3, 8, 17, 257}) {
            for (int k : {1, std::min(3, vocab), vocab})
                test(3, vocab, k, distribution);
        }
        test(8, 8192, 10, distribution);
        test(1, 50257, 100, distribution);
    }
    test(32, 50257, 50, 0);
    std::cout << "PASS: random, ties, equal/negative scores, padding, batches, full "
                 "vocabulary, input preservation\n";
}
