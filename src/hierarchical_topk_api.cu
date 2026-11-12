#include "topk.cuh"

extern "C" {
std::size_t hierarchical_topk_workspace_size(int batch, int vocab, int k) {
    return topk::hierarchical_workspace_elements(batch, vocab, k);
}

int hierarchical_topk_run(const float* input, int batch, int vocab, int k,
                          float* output_scores, int* output_ids,
                          float* work_scores_a, int* work_ids_a,
                          float* work_scores_b, int* work_ids_b,
                          std::size_t capacity, void* stream) {
    const topk::HierarchicalWorkspace workspace{
        work_scores_a, work_ids_a, work_scores_b, work_ids_b, capacity};
    return static_cast<int>(topk::hierarchical_topk(
        input, batch, vocab, k, output_scores, output_ids, workspace,
        reinterpret_cast<cudaStream_t>(stream)));
}

const char* hierarchical_topk_error_string(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}
}
