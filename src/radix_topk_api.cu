#include "topk.cuh"

extern "C" {
int radix_topk_workspace_size(int batch, int vocab, int k,
                              std::size_t* bytes, void* stream) {
    return static_cast<int>(topk::radix_topk_workspace_bytes(
        batch, vocab, k, bytes, reinterpret_cast<cudaStream_t>(stream)));
}

int radix_topk_run(const float* input, int batch, int vocab, int k,
                   float* output_scores, int* output_ids,
                   void* workspace, std::size_t workspace_bytes, void* stream) {
    return static_cast<int>(topk::radix_topk(
        input, batch, vocab, k, output_scores, output_ids,
        {workspace, workspace_bytes}, reinterpret_cast<cudaStream_t>(stream)));
}

const char* radix_topk_error_string(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}
}
