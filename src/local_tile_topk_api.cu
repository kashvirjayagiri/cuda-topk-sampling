#include "topk.cuh"

extern "C" {
int local_tile_count(int vocab) {
    return topk::local_tile_count(vocab);
}

int local_tile_topk_run(const float* input, int batch, int vocab, int k,
                        float* output_scores, int* output_ids, void* stream) {
    return static_cast<int>(topk::local_tile_topk(
        input, batch, vocab, k, output_scores, output_ids,
        reinterpret_cast<cudaStream_t>(stream)));
}

const char* local_tile_topk_error_string(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}
}
