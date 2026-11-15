#include "hierarchical_topk_sampling.cuh"

extern "C" {
std::size_t hierarchical_topk_sampling_workspace_size(int batch, int vocab, int k) {
    return topk::hierarchical_workspace_elements(batch, vocab, k);
}

int hierarchical_topk_sampling_run(
    const float* logits, const float* randoms, int batch, int vocab, int k,
    float temperature, float* selected_scores, int* selected_ids,
    float* work_scores_a, int* work_ids_a, float* work_scores_b, int* work_ids_b,
    std::size_t work_capacity, float* probabilities, float* cdf,
    int* sampled_ids, void* stream) {
    const topk::HierarchicalTopKSamplingBuffers buffers{
        {work_scores_a, work_ids_a, work_scores_b, work_ids_b, work_capacity},
        selected_scores, selected_ids, probabilities, cdf, sampled_ids};
    return static_cast<int>(topk::hierarchical_topk_sample(
        logits, randoms, batch, vocab, k, temperature, buffers,
        reinterpret_cast<cudaStream_t>(stream)));
}

const char* hierarchical_topk_sampling_error_string(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}
}
