// read test input from python and run sampler cu on same input
// check driver not bench no softmax here

// DONE BY JOEL SEDILLO

#include "sampler.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(1);
    }
}

void print_floats(const std::vector<float>& values) {
    for (std::size_t i = 0; i < values.size(); i++) {
        if (i != 0) std::cout << ' ';
        std::cout << values[i];
    }
    std::cout << '\n';
}

void print_ids(const std::vector<std::int32_t>& values) {
    for (std::size_t i = 0; i < values.size(); i++) {
        if (i != 0) std::cout << ' ';
        std::cout << values[i];
    }
    std::cout << '\n';
}

int main() {
    // nine sig digits keep fp32 in text
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    int B, k;
    float temperature;
    // one job handles all tests
    while (std::cin >> B >> k >> temperature) {
        if (B < 1 || k < 1 || !std::isfinite(temperature) || temperature <= 0) {
            std::fprintf(stderr, "Invalid dimensions or temperature\n");
            return 1;
        }
        std::size_t count = (std::size_t)B * k;
        if (count > 16777216) {
            std::fprintf(stderr, "Test case exceeds this driver's input limit\n");
            return 1;
        }
        std::vector<float> scores(count), randoms(B), probabilities(count), cdf(count);
        std::vector<std::int32_t> ids(count), sampled(B);
        for (std::size_t i = 0; i < count; i++) {
            if (!(std::cin >> scores[i]) || !std::isfinite(scores[i])) return 1;
        }
        for (std::size_t i = 0; i < count; i++) {
            if (!(std::cin >> ids[i]) || ids[i] < 0) return 1;
        }
        for (int row = 0; row < B; row++) {
            if (!(std::cin >> randoms[row]) || !std::isfinite(randoms[row])
                || randoms[row] < 0 || randoms[row] >= 1) return 1;
        }

        float *device_scores, *device_randoms, *device_probabilities, *device_cdf;
        std::int32_t *device_ids, *device_sampled;
        check_cuda(cudaMalloc((void**)&device_scores, count*sizeof(float)), "allocate scores");
        check_cuda(cudaMalloc((void**)&device_ids, count*sizeof(std::int32_t)), "allocate IDs");
        check_cuda(cudaMalloc((void**)&device_randoms, B*sizeof(float)), "allocate randoms");
        check_cuda(cudaMalloc((void**)&device_probabilities, count*sizeof(float)), "allocate probabilities");
        check_cuda(cudaMalloc((void**)&device_cdf, count*sizeof(float)), "allocate CDF");
        check_cuda(cudaMalloc((void**)&device_sampled, B*sizeof(std::int32_t)), "allocate output IDs");
        check_cuda(cudaMemcpy(device_scores, scores.data(), count*sizeof(float), cudaMemcpyHostToDevice), "upload scores");
        check_cuda(cudaMemcpy(device_ids, ids.data(), count*sizeof(std::int32_t), cudaMemcpyHostToDevice), "upload IDs");
        check_cuda(cudaMemcpy(device_randoms, randoms.data(), B*sizeof(float), cudaMemcpyHostToDevice), "upload randoms");

        for (int mode = 0; mode < 2; mode++) {
            cudaError_t status;
            if (mode == 0) {
                status = jjk::sample_serial(device_scores, device_ids, device_randoms,
                    B, k, temperature, device_probabilities, device_cdf, device_sampled, 0);
            } else {
                status = jjk::sample_parallel(device_scores, device_ids, device_randoms,
                    B, k, temperature, device_probabilities, device_cdf, device_sampled, 0);
            }
            check_cuda(status, "launch sampler");
            check_cuda(cudaDeviceSynchronize(), "finish sampler");
            check_cuda(cudaMemcpy(probabilities.data(), device_probabilities, count*sizeof(float), cudaMemcpyDeviceToHost), "read probabilities");
            check_cuda(cudaMemcpy(cdf.data(), device_cdf, count*sizeof(float), cudaMemcpyDeviceToHost), "read CDF");
            check_cuda(cudaMemcpy(sampled.data(), device_sampled, B*sizeof(std::int32_t), cudaMemcpyDeviceToHost), "read output IDs");
            std::cout << (mode == 0 ? "serial" : "parallel") << '\n';
            print_floats(probabilities);
            print_floats(cdf);
            print_ids(sampled);
        }
        std::cout << "END\n";
        check_cuda(cudaFree(device_scores), "free scores");
        check_cuda(cudaFree(device_ids), "free IDs");
        check_cuda(cudaFree(device_randoms), "free randoms");
        check_cuda(cudaFree(device_probabilities), "free probabilities");
        check_cuda(cudaFree(device_cdf), "free CDF");
        check_cuda(cudaFree(device_sampled), "free output IDs");
    }
    if (!std::cin.eof()) {
        std::fprintf(stderr, "Malformed input header\n");
        return 1;
    }
    return 0;
}
