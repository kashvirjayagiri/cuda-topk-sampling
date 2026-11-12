#include <climits>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

#include "topk.cuh"

namespace topk {
namespace {

constexpr int kThreads = 256;
constexpr int kChunk = 1024;
constexpr int kBins = 256;

struct RowState {
    std::uint32_t prefix;
    int rank;
    int greater;
};

struct Plan {
    int items;
    int selected;
    int chunks;
    int blocks;
    std::size_t partial_offset;
    std::size_t greater_flags_offset;
    std::size_t equal_flags_offset;
    std::size_t greater_scan_offset;
    std::size_t equal_scan_offset;
    std::size_t keys_in_offset;
    std::size_t keys_out_offset;
    std::size_t state_offset;
    std::size_t segments_offset;
    std::size_t scratch_offset;
    std::size_t scratch_bytes;
    std::size_t bytes;
};

bool reserve(std::size_t count, std::size_t element_bytes, std::size_t alignment,
             std::size_t* cursor, std::size_t* offset) {
    const auto max = std::numeric_limits<std::size_t>::max();
    if (count > max / element_bytes || *cursor > max - (alignment - 1))
        return false;
    const auto aligned = (*cursor + alignment - 1) & ~(alignment - 1);
    const auto bytes = count * element_bytes;
    if (aligned > max - bytes)
        return false;
    *offset = aligned;
    *cursor = aligned + bytes;
    return true;
}

cudaError_t make_plan(int batch, int vocab, int k, cudaStream_t stream, Plan* plan) {
    if (!plan || batch <= 0 || vocab <= 0 || k <= 0 || k > vocab)
        return cudaErrorInvalidValue;
    const auto items = static_cast<std::size_t>(batch) * vocab;
    const auto selected = static_cast<std::size_t>(batch) * k;
    const int chunks = 1 + (vocab - 1) / kChunk;
    const auto blocks = static_cast<std::size_t>(batch) * chunks;
    if (items > INT_MAX || selected > INT_MAX || blocks > INT_MAX)
        return cudaErrorInvalidValue;

    std::size_t scan_bytes = 0;
    auto error = cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, static_cast<const int*>(nullptr),
        static_cast<int*>(nullptr), static_cast<int>(items), stream);
    if (error != cudaSuccess)
        return error;
    std::size_t sort_bytes = 0;
    error = cub::DeviceSegmentedRadixSort::SortKeysDescending(
        nullptr, sort_bytes, static_cast<const unsigned long long*>(nullptr),
        static_cast<unsigned long long*>(nullptr), static_cast<int>(selected), batch,
        static_cast<const int*>(nullptr), static_cast<const int*>(nullptr),
        0, 64, stream);
    if (error != cudaSuccess)
        return error;

    plan->items = static_cast<int>(items);
    plan->selected = static_cast<int>(selected);
    plan->chunks = chunks;
    plan->blocks = static_cast<int>(blocks);
    plan->scratch_bytes = scan_bytes > sort_bytes ? scan_bytes : sort_bytes;
    if (plan->scratch_bytes == 0)
        plan->scratch_bytes = 1;

    // B=32,V=131072,k=100: partial histograms 4 MiB; two flags plus two
    // scans 64 MiB; two selected-key arrays 50 KiB; row state/offsets 516 B.
    // Add the queried maximum CUB scratch and alignment for the exact total.
    std::size_t cursor = 0;
    if (!reserve(blocks * kBins, sizeof(std::uint32_t), alignof(std::uint32_t),
                 &cursor, &plan->partial_offset) ||
        !reserve(items, sizeof(int), alignof(int), &cursor, &plan->greater_flags_offset) ||
        !reserve(items, sizeof(int), alignof(int), &cursor, &plan->equal_flags_offset) ||
        !reserve(items, sizeof(int), alignof(int), &cursor, &plan->greater_scan_offset) ||
        !reserve(items, sizeof(int), alignof(int), &cursor, &plan->equal_scan_offset) ||
        !reserve(selected, sizeof(unsigned long long), alignof(unsigned long long),
                 &cursor, &plan->keys_in_offset) ||
        !reserve(selected, sizeof(unsigned long long), alignof(unsigned long long),
                 &cursor, &plan->keys_out_offset) ||
        !reserve(batch, sizeof(RowState), alignof(RowState), &cursor, &plan->state_offset) ||
        !reserve(static_cast<std::size_t>(batch) + 1, sizeof(int), alignof(int),
                 &cursor, &plan->segments_offset) ||
        !reserve(plan->scratch_bytes, 1, 16, &cursor, &plan->scratch_offset))
        return cudaErrorInvalidValue;
    plan->bytes = cursor;
    return cudaSuccess;
}

__device__ __forceinline__ std::uint32_t score_key(float score) {
    std::uint32_t bits = __float_as_uint(score);
    if ((bits & 0x7fffffffU) == 0)
        bits = 0;  // Both signed zeros compare equal; original bits stay in input.
    return (bits & 0x80000000U) ? ~bits : bits ^ 0x80000000U;
}

__global__ void init_state_kernel(RowState* state, int batch, int k) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < static_cast<std::size_t>(batch))
        state[row] = RowState{0U, k, 0};
}

__global__ void histogram_kernel(const float* input, int vocab, int chunks,
                                 int shift, const RowState* state,
                                 std::uint32_t* partial) {
    const int row = blockIdx.x / chunks;
    const int chunk = blockIdx.x % chunks;
    const int lane = threadIdx.x;
    __shared__ std::uint32_t histogram[kBins];
    histogram[lane] = 0;
    __syncthreads();

    for (int offset = lane; offset < kChunk; offset += kThreads) {
        const int token = chunk * kChunk + offset;
        if (token < vocab) {
            const auto key = score_key(input[static_cast<std::size_t>(row) * vocab + token]);
            if (shift == 24 || (key >> (shift + 8)) == (state[row].prefix >> (shift + 8)))
                atomicAdd(&histogram[(key >> shift) & 0xffU], 1U);
        }
    }
    __syncthreads();
    partial[static_cast<std::size_t>(blockIdx.x) * kBins + lane] = histogram[lane];
}

__global__ void choose_bucket_kernel(const std::uint32_t* partial, int chunks,
                                     int shift, RowState* state) {
    const int row = blockIdx.x;
    const int bin = threadIdx.x;
    std::uint32_t count = 0;
    for (int chunk = 0; chunk < chunks; ++chunk)
        count += partial[(static_cast<std::size_t>(row) * chunks + chunk) * kBins + bin];
    __shared__ std::uint32_t histogram[kBins];
    histogram[bin] = count;
    __syncthreads();

    if (bin == 0) {
        RowState current = state[row];
        int greater_this_pass = 0;
        for (int bucket = kBins - 1; bucket >= 0; --bucket) {
            if (current.rank > static_cast<int>(histogram[bucket])) {
                current.rank -= static_cast<int>(histogram[bucket]);
                greater_this_pass += static_cast<int>(histogram[bucket]);
            } else {
                current.prefix |= static_cast<std::uint32_t>(bucket) << shift;
                break;
            }
        }
        current.greater += greater_this_pass;
        state[row] = current;
    }
}

__global__ void flags_kernel(const float* input, int vocab, int items,
                             const RowState* state, int* greater, int* equal) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < static_cast<std::size_t>(items)) {
        const auto key = score_key(input[index]);
        const auto threshold = state[index / vocab].prefix;
        greater[index] = key > threshold;
        equal[index] = key == threshold;
    }
}

__global__ void compact_kernel(const float* input, int vocab, int items, int k,
                               const RowState* state, const int* greater,
                               const int* equal, const int* greater_scan,
                               const int* equal_scan, unsigned long long* keys) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(items))
        return;
    const int row = static_cast<int>(index / vocab);
    const int token = static_cast<int>(index % vocab);
    int destination = -1;
    if (greater[index]) {
        destination = greater_scan[index] - greater_scan[row * vocab];
    } else if (equal[index]) {
        const int equal_rank = equal_scan[index] - equal_scan[row * vocab];
        if (equal_rank < state[row].rank)
            destination = state[row].greater + equal_rank;
    }
    if (destination >= 0) {
        const auto key = score_key(input[index]);
        keys[static_cast<std::size_t>(row) * k + destination] =
            (static_cast<unsigned long long>(key) << 32) |
            (0xffffffffULL - static_cast<std::uint32_t>(token));
    }
}

__global__ void segment_offsets_kernel(int* offsets, int batch, int k) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index <= static_cast<std::size_t>(batch))
        offsets[index] = static_cast<int>(index * k);
}

__global__ void gather_kernel(const float* input, int vocab, int k, int selected,
                              const unsigned long long* keys, float* scores, int* ids) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < static_cast<std::size_t>(selected)) {
        const int row = static_cast<int>(index / k);
        const int token = static_cast<int>(~static_cast<std::uint32_t>(keys[index]));
        ids[index] = token;
        scores[index] = input[static_cast<std::size_t>(row) * vocab + token];
    }
}

}  // namespace

cudaError_t radix_topk_workspace_bytes(int batch, int vocab, int k,
                                      std::size_t* bytes, cudaStream_t stream) {
    if (!bytes)
        return cudaErrorInvalidValue;
    *bytes = 0;
    Plan plan{};
    const auto error = make_plan(batch, vocab, k, stream, &plan);
    if (error == cudaSuccess)
        *bytes = plan.bytes;
    return error;
}

cudaError_t radix_topk(const float* input, int batch, int vocab, int k,
                       float* output_values, int* output_ids,
                       RadixWorkspace workspace, cudaStream_t stream) {
    if (!input || !output_values || !output_ids || !workspace.data)
        return cudaErrorInvalidValue;
    Plan plan{};
    auto error = make_plan(batch, vocab, k, stream, &plan);
    if (error != cudaSuccess)
        return error;
    if (workspace.bytes < plan.bytes)
        return cudaErrorInvalidValue;

    auto* base = static_cast<unsigned char*>(workspace.data);
    auto* partial = reinterpret_cast<std::uint32_t*>(base + plan.partial_offset);
    auto* greater_flags = reinterpret_cast<int*>(base + plan.greater_flags_offset);
    auto* equal_flags = reinterpret_cast<int*>(base + plan.equal_flags_offset);
    auto* greater_scan = reinterpret_cast<int*>(base + plan.greater_scan_offset);
    auto* equal_scan = reinterpret_cast<int*>(base + plan.equal_scan_offset);
    auto* keys_in = reinterpret_cast<unsigned long long*>(base + plan.keys_in_offset);
    auto* keys_out = reinterpret_cast<unsigned long long*>(base + plan.keys_out_offset);
    auto* state = reinterpret_cast<RowState*>(base + plan.state_offset);
    auto* offsets = reinterpret_cast<int*>(base + plan.segments_offset);
    void* scratch = base + plan.scratch_offset;

    init_state_kernel<<<(batch - 1) / kThreads + 1, kThreads, 0, stream>>>(state, batch, k);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return error;
    for (int shift : {24, 16, 8, 0}) {
        histogram_kernel<<<plan.blocks, kThreads, 0, stream>>>(
            input, vocab, plan.chunks, shift, state, partial);
        error = cudaGetLastError();
        if (error != cudaSuccess)
            return error;
        choose_bucket_kernel<<<batch, kThreads, 0, stream>>>(partial, plan.chunks, shift, state);
        error = cudaGetLastError();
        if (error != cudaSuccess)
            return error;
    }

    const int item_blocks = (plan.items - 1) / kThreads + 1;
    flags_kernel<<<item_blocks, kThreads, 0, stream>>>(
        input, vocab, plan.items, state, greater_flags, equal_flags);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return error;
    std::size_t scratch_bytes = plan.scratch_bytes;
    error = cub::DeviceScan::ExclusiveSum(
        scratch, scratch_bytes, greater_flags, greater_scan, plan.items, stream);
    if (error != cudaSuccess)
        return error;
    scratch_bytes = plan.scratch_bytes;
    error = cub::DeviceScan::ExclusiveSum(
        scratch, scratch_bytes, equal_flags, equal_scan, plan.items, stream);
    if (error != cudaSuccess)
        return error;

    compact_kernel<<<item_blocks, kThreads, 0, stream>>>(
        input, vocab, plan.items, k, state, greater_flags, equal_flags,
        greater_scan, equal_scan, keys_in);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return error;
    segment_offsets_kernel<<<batch / kThreads + 1, kThreads, 0, stream>>>(
        offsets, batch, k);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return error;

    scratch_bytes = plan.scratch_bytes;
    error = cub::DeviceSegmentedRadixSort::SortKeysDescending(
        scratch, scratch_bytes, keys_in, keys_out, plan.selected, batch,
        offsets, offsets + 1, 0, 64, stream);
    if (error != cudaSuccess)
        return error;
    gather_kernel<<<(plan.selected - 1) / kThreads + 1, kThreads, 0, stream>>>(
        input, vocab, k, plan.selected, keys_out, output_values, output_ids);
    return cudaGetLastError();
}

}  // namespace topk
