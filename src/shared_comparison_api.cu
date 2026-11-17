// Shared runner only: reuse the team's existing selectors and integrations.
#include "full_sort_sampling.cuh"
#include "hierarchical_topk_sampling.cuh"
#include "topk.cuh"

static_assert(sizeof(int) == sizeof(std::int32_t), "Sampler IDs require int32");

extern "C" {
// method 0 = full sort; method 1 = hierarchical; method 2 = radix.
// Element count is per workspace buffer for methods 0 and 1.
std::size_t comparison_workspace_elements(int method, int B, int V, int k) {
    if (B < 1 || k < 1 || k > V) return 0;
    if (method == 0) return topk::full_sort_workspace_elements(B, V);
    if (method == 1) return topk::hierarchical_workspace_elements(B, V, k);
    return 0;
}

int comparison_radix_workspace_bytes(int B, int V, int k,
                                     std::size_t* bytes, void* stream_pointer) {
    return static_cast<int>(topk::radix_topk_workspace_bytes(
        B, V, k, bytes, reinterpret_cast<cudaStream_t>(stream_pointer)));
}

int comparison_select(int method, const float* logits, int B, int V, int k,
    float* scores, int* ids, float* wa, int* ia, float* wb, int* ib,
    std::size_t capacity, void* stream_pointer) {
    const auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    if (method == 0)
        return static_cast<int>(topk::full_sort(logits, B, V, k, scores, ids,
                                               {wa, ia, capacity}, stream));
    if (method == 1)
        return static_cast<int>(topk::hierarchical_topk(logits, B, V, k, scores, ids,
                                                       {wa, ia, wb, ib, capacity}, stream));
    if (method == 2)
        return static_cast<int>(topk::radix_topk(logits, B, V, k, scores, ids,
                                                {wa, capacity}, stream));
    return static_cast<int>(cudaErrorInvalidValue);
}

int comparison_sample(const float* scores, const int* ids, const float* randoms,
    int B, int k, float temperature, float* probabilities, float* cdf,
    int* sampled_ids, void* stream_pointer) {
    return static_cast<int>(jjk::sample_parallel(scores, ids, randoms, B, k,
        temperature, probabilities, cdf, sampled_ids,
        reinterpret_cast<cudaStream_t>(stream_pointer)));
}

int comparison_pipeline(int method, const float* logits, const float* randoms,
    int B, int V, int k, float temperature, float* scores, int* ids,
    float* wa, int* ia, float* wb, int* ib, std::size_t capacity,
    float* probabilities, float* cdf, int* sampled_ids, void* stream_pointer) {
    const auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    if (method == 0) {
        const topk::FullSortSamplingBuffers buffers{
            {wa, ia, capacity}, scores, ids, probabilities, cdf, sampled_ids};
        return static_cast<int>(topk::full_sort_sample(
            logits, randoms, B, V, k, temperature, buffers, stream));
    }
    if (method == 1) {
        const topk::HierarchicalTopKSamplingBuffers buffers{
            {wa, ia, wb, ib, capacity}, scores, ids, probabilities, cdf, sampled_ids};
        return static_cast<int>(topk::hierarchical_topk_sample(
            logits, randoms, B, V, k, temperature, buffers, stream));
    }
    if (method == 2) {
        const auto status = topk::radix_topk(logits, B, V, k, scores, ids,
                                             {wa, capacity}, stream);
        if (status != cudaSuccess) return static_cast<int>(status);
        return static_cast<int>(jjk::sample_parallel(
            scores, ids, randoms, B, k, temperature, probabilities, cdf,
            sampled_ids, stream));
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

const char* comparison_error_string(int status) {
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}

int comparison_versions(int* runtime, int* driver) {
    if (!runtime || !driver) return static_cast<int>(cudaErrorInvalidValue);
    auto status = cudaRuntimeGetVersion(runtime);
    if (status == cudaSuccess) status = cudaDriverGetVersion(driver);
    return static_cast<int>(status);
}

#define JJK_STRING_INNER(x) #x
#define JJK_STRING(x) JJK_STRING_INNER(x)
const char* comparison_nvcc_version() {
    return JJK_STRING(__CUDACC_VER_MAJOR__) "." JJK_STRING(__CUDACC_VER_MINOR__)
           "." JJK_STRING(__CUDACC_VER_BUILD__);
}
}
