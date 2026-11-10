// A small C++/CUDA example following the lectures' host -> device -> host flow.
// CUDA API spellings are implementation details beyond the slides' pseudocode.
#include "sampler.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

// error handling is supporting code, not a parallel algorithm
void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(1);
    }
}

int main() {
    // host code prepares data for the GPU kernel.
    const int B = 2;
    const int k = 3;
    const float temperature = 1.0f;
    float scores[B*k] = {3.0f, 2.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    int token_ids[B*k] = {17, 42, 90, 5, 6, 7};
    float random_values[B] = {0.8f, 0.5f};
    float probabilities[B*k];
    float cdf[B*k];
    int sampled_ids[B];
    int expected_ids[B] = {42, 6};

    float* device_scores;
    int* device_ids;
    float* device_randoms;
    float* device_probabilities;
    float* device_cdf;
    int* device_sampled_ids;

    // allocate GPU memory
    check_cuda(cudaMalloc((void**)&device_scores, sizeof(scores)), "allocate scores");
    check_cuda(cudaMalloc((void**)&device_ids, sizeof(token_ids)), "allocate IDs");
    check_cuda(cudaMalloc((void**)&device_randoms, sizeof(random_values)), "allocate randoms");
    check_cuda(cudaMalloc((void**)&device_probabilities, sizeof(probabilities)), "allocate probabilities");
    check_cuda(cudaMalloc((void**)&device_cdf, sizeof(cdf)), "allocate CDF");
    check_cuda(cudaMalloc((void**)&device_sampled_ids, sizeof(sampled_ids)), "allocate output IDs");

    // copy input data from host (CPU) to device (GPU)
    check_cuda(cudaMemcpy(device_scores, scores, sizeof(scores), cudaMemcpyHostToDevice), "copy scores");
    check_cuda(cudaMemcpy(device_ids, token_ids, sizeof(token_ids), cudaMemcpyHostToDevice), "copy IDs");
    check_cuda(cudaMemcpy(device_randoms, random_values, sizeof(random_values), cudaMemcpyHostToDevice), "copy randoms");

    for (int mode = 0; mode < 2; mode++) {
        // the wrappers in sampler.cu launch <<<1,1>>> (serial) or
        // <<<B,256>>> (parallel), on the stream passed as the last argument
        cudaError_t status;
        if (mode == 0) {
            status = jjk::sample_serial(device_scores, device_ids, device_randoms,
                B, k, temperature, device_probabilities, device_cdf,
                device_sampled_ids, 0);
        } else {
            status = jjk::sample_parallel(device_scores, device_ids, device_randoms,
                B, k, temperature, device_probabilities, device_cdf,
                device_sampled_ids, 0);
        }
        check_cuda(status, "launch sampler");
        check_cuda(cudaDeviceSynchronize(), "finish sampler");

        // copy results back only AFTER computation. Not part of GPU timing
        check_cuda(cudaMemcpy(probabilities, device_probabilities, sizeof(probabilities), cudaMemcpyDeviceToHost), "read probabilities");
        check_cuda(cudaMemcpy(cdf, device_cdf, sizeof(cdf), cudaMemcpyDeviceToHost), "read CDF");
        check_cuda(cudaMemcpy(sampled_ids, device_sampled_ids, sizeof(sampled_ids), cudaMemcpyDeviceToHost), "read output IDs");

        std::printf("%s\n", mode == 0 ? "Serial CUDA:" : "Parallel CUDA:");
        for (int row = 0; row < B; row++) {
            std::printf("  row %d: token=%d, probabilities=", row, sampled_ids[row]);
            float total = 0.0f;
            for (int i = 0; i < k; i++) {
                std::printf(" %.6f", probabilities[row*k+i]);
                total += probabilities[row*k+i];
            }
            std::printf("\n");
            if (sampled_ids[row] != expected_ids[row] || std::fabs(total-1.0f) > 1e-5f
                || cdf[row*k+k-1] != 1.0f) {
                std::fprintf(stderr, "FAIL: unexpected result in row %d\n", row);
                return 1;
            }
        }
    }

    // device memory lifetime ends here 
    check_cuda(cudaFree(device_scores), "free scores");
    check_cuda(cudaFree(device_ids), "free IDs");
    check_cuda(cudaFree(device_randoms), "free randoms");
    check_cuda(cudaFree(device_probabilities), "free probabilities");
    check_cuda(cudaFree(device_cdf), "free CDF");
    check_cuda(cudaFree(device_sampled_ids), "free output IDs");
    std::puts("PASS: both samplers returned the expected original token IDs.");
    return 0;
}
