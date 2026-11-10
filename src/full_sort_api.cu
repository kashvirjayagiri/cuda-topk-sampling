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

// Timing stays in C++ so Python/ctypes work is outside CUDA event intervals.
// host_milliseconds is a CPU buffer of `repetitions` floats owned by Python.
int full_sort_benchmark(
    const float* input, int batch_size, int vocab_size, int k,
    float* output_scores, int* output_ids,
    float* work_scores, int* work_ids, std::size_t capacity,
    void* stream_pointer, int warmup, int repetitions, float* host_milliseconds) {
    if (warmup < 0 || repetitions <= 0 || !host_milliseconds) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    const topk::FullSortWorkspace workspace{work_scores, work_ids, capacity};
    auto run = [&]() {
        return topk::full_sort(input, batch_size, vocab_size, k,
                              output_scores, output_ids, workspace, stream);
    };

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    auto cleanup = [&](cudaError_t error) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        return static_cast<int>(error);
    };
    auto error = cudaEventCreate(&start);
    if (error != cudaSuccess) return cleanup(error);
    error = cudaEventCreate(&stop);
    if (error != cudaSuccess) return cleanup(error);

    for (int iteration = 0; iteration < warmup; ++iteration) {
        error = run();
        if (error != cudaSuccess) return cleanup(error);
    }
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) return cleanup(error);

    for (int iteration = 0; iteration < repetitions; ++iteration) {
        error = cudaEventRecord(start, stream);
        if (error != cudaSuccess) return cleanup(error);
        error = run();
        if (error != cudaSuccess) return cleanup(error);
        error = cudaEventRecord(stop, stream);
        if (error != cudaSuccess) return cleanup(error);
        error = cudaEventSynchronize(stop);
        if (error != cudaSuccess) return cleanup(error);
        error = cudaEventElapsedTime(&host_milliseconds[iteration], start, stop);
        if (error != cudaSuccess) return cleanup(error);
    }
    return cleanup(cudaSuccess);
}

const char* full_sort_error_string(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}
}
