#include "rejection_sampler.cuh"

extern "C" {
int rejection_sample_topk_run(const float* logits, int batch, int vocab, int k,
                              float temperature, unsigned long long seed,
                              unsigned long long draw_index, int* sampled_ids,
                              int* attempts, void* stream) {
    return static_cast<int>(topk::rejection_sample_topk(
        logits, batch, vocab, k, temperature, seed, draw_index,
        sampled_ids, attempts, reinterpret_cast<cudaStream_t>(stream)));
}

const char* rejection_sample_topk_error_string(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}
}
