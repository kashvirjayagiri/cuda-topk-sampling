// Thin C interface so Python can call the existing sorter through ctypes.
// No PyTorch C++ dependency: Python owns all device tensors and their lifetime.
#include "topk.cuh"

extern "C" {
std::size_t full_sort_workspace_size(int batch_size, int vocab_size) {
    return topk::full_sort_workspace_elements(batch_size, vocab_size);
}

int full_sort_run(const float* input, int batch_size, int vocab_size, int k,
                  float* output_scores, int* output_ids,
                  float* work_scores, int* work_ids, std::size_t capacity,
                  void* stream) {
    const topk::FullSortWorkspace workspace{work_scores, work_ids, capacity};
    return static_cast<int>(topk::full_sort(
        input, batch_size, vocab_size, k, output_scores, output_ids,
        workspace, reinterpret_cast<cudaStream_t>(stream)));
}

const char* full_sort_error_string(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}
}
