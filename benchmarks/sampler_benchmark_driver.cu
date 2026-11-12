// cuda event sampler benchmark
// link with src/sampler.cu
// python feeds input and refs
#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(1);
    }
}

struct DeviceBuffers {
    int B, k;
    float *scores, *randoms, *probabilities, *cdf;
    std::int32_t *ids, *sampled;
};

void launch_sampler(DeviceBuffers& d, int mode, float temperature, cudaStream_t stream) {
    cudaError_t status;
    if (mode == 0) {
        status = jjk::sample_serial(d.scores, d.ids, d.randoms, d.B, d.k,
            temperature, d.probabilities, d.cdf, d.sampled, stream);
    } else {
        status = jjk::sample_parallel(d.scores, d.ids, d.randoms, d.B, d.k,
            temperature, d.probabilities, d.cdf, d.sampled, stream);
    }
    check_cuda(status, "launch sampler");
}

float measure(DeviceBuffers& d, int mode, float temperature, int launches,
    cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
    check_cuda(cudaEventRecord(start, stream), "record start");
    for (int i = 0; i < launches; i++) launch_sampler(d, mode, temperature, stream);
    check_cuda(cudaEventRecord(stop, stream), "record stop");
    check_cuda(cudaEventSynchronize(stop), "wait for stop");
    float milliseconds;
    check_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "read elapsed time");
    return milliseconds * 1000.0f / launches;
}

bool close_enough(double value, double reference) {
    return std::isfinite(value) && std::fabs(value-reference) <= 1e-5 + 1e-5*std::fabs(reference);
}

bool validate(const std::vector<float>& p, const std::vector<float>& cdf,
    const std::vector<std::int32_t>& sampled, const std::vector<std::int32_t>& ids,
    const std::vector<float>& u, const std::vector<double>& expected_p,
    const std::vector<double>& expected_cdf, const std::vector<std::int32_t>& expected_ids,
    int B, int k, double& p_error, double& cdf_error) {
    p_error = 0.0; cdf_error = 0.0;
    for (int row = 0; row < B; row++) {
        std::size_t base = (std::size_t)row*k;
        double sum = 0.0;
        int first = -1;
        for (int i = 0; i < k; i++) {
            std::size_t index = base+i;
            if (p[index] < 0 || p[index] > 1 || cdf[index] < 0 || cdf[index] > 1
                || !close_enough(p[index], expected_p[index])
                || !close_enough(cdf[index], expected_cdf[index])
                || (i > 0 && cdf[index] < cdf[index-1]-1e-7f)) {
                std::fprintf(stderr, "Validation failed: row %d candidate %d; p=%.9g expected=%.17g, CDF=%.9g expected=%.17g\n",
                    row, i, p[index], expected_p[index], cdf[index], expected_cdf[index]);
                return false;
            }
            sum += p[index];
            p_error = std::max(p_error, std::fabs(p[index]-expected_p[index]));
            cdf_error = std::max(cdf_error, std::fabs(cdf[index]-expected_cdf[index]));
            if (first < 0 && cdf[index] > u[row]) first = i;
        }
        if (!close_enough(sum, 1.0) || cdf[base+k-1] != 1.0f || first < 0
            || sampled[row] != ids[base+first] || sampled[row] != expected_ids[row]) {
            double distance = 1.0;
            for (int i = 0; i < k; i++) distance = std::min(distance, std::fabs(expected_cdf[base+i]-u[row]));
            std::fprintf(stderr, "Validation failed: row %d; GPU token=%d, Python token=%d; nearest reference CDF boundary=%.9g. Inspect rounding if near a boundary.\n",
                row, sampled[row], expected_ids[row], distance);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Run through benchmark_sampler.py (arguments: warmup repeats launches_per_sample)\n");
        return 1;
    }
    char *end1, *end2, *end3;
    long warmup_value = std::strtol(argv[1], &end1, 10);
    long repeats_value = std::strtol(argv[2], &end2, 10);
    long launches_value = std::strtol(argv[3], &end3, 10);
    if (*end1 || *end2 || *end3 || warmup_value < 0 || warmup_value > 100000
        || repeats_value < 1 || repeats_value > 100000 || launches_value < 1 || launches_value > 100000) return 1;
    int warmup = (int)warmup_value, repeats = (int)repeats_value, launches = (int)launches_value;

    int device, runtime, driver;
    cudaDeviceProp properties;
    check_cuda(cudaGetDevice(&device), "get current device");
    check_cuda(cudaGetDeviceProperties(&properties, device), "get GPU information");
    check_cuda(cudaRuntimeGetVersion(&runtime), "get CUDA runtime version");
    check_cuda(cudaDriverGetVersion(&driver), "get CUDA driver version");
    std::cout << std::setprecision(17);
    std::cout << "DEVICE " << properties.name << '\n';
    std::cout << "COMPUTE " << properties.major << '.' << properties.minor << '\n';
    std::cout << "CUDA_RUNTIME " << runtime << '\n';
    std::cout << "CUDA_DRIVER " << driver << '\n';
    std::cout << "NVCC " << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '.' << __CUDACC_VER_BUILD__ << '\n';

    cudaStream_t stream;
    cudaEvent_t start, stop;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    int B, k, case_number = 0;
    double input_temperature;
    while (std::cin >> B >> k >> input_temperature) {
        float temperature = (float)input_temperature;
        if (B < 1 || k < 1 || !std::isfinite(temperature) || temperature <= 0) return 1;
        std::size_t count = (std::size_t)B*k;
        if (count > 16777216) { std::fprintf(stderr, "Input exceeds benchmark driver limit\n"); return 1; }
        std::size_t bytes = count*sizeof(float);
        std::vector<float> scores(count), randoms(B), p(count), cdf(count);
        std::vector<std::int32_t> ids(count), sampled(B), expected_ids(B);
        std::vector<double> expected_p(count), expected_cdf(count);
        for (std::size_t i = 0; i < count; i++) {
            double value;
            if (!(std::cin >> value)) return 1;
            scores[i] = (float)value;
            if (!std::isfinite(scores[i])) return 1;
        }
        for (std::size_t i = 0; i < count; i++) if (!(std::cin >> ids[i]) || ids[i] < 0) return 1;
        for (int row = 0; row < B; row++) {
            if (!(std::cin >> randoms[row]) || !std::isfinite(randoms[row]) || randoms[row] < 0 || randoms[row] >= 1) return 1;
        }
        for (std::size_t i = 0; i < count; i++) if (!(std::cin >> expected_p[i]) || !std::isfinite(expected_p[i])) return 1;
        for (std::size_t i = 0; i < count; i++) if (!(std::cin >> expected_cdf[i]) || !std::isfinite(expected_cdf[i])) return 1;
        for (int row = 0; row < B; row++) if (!(std::cin >> expected_ids[row])) return 1;

        DeviceBuffers d;
        d.B = B; d.k = k;
        check_cuda(cudaMalloc((void**)&d.scores, bytes), "allocate scores");
        check_cuda(cudaMalloc((void**)&d.ids, count*sizeof(std::int32_t)), "allocate IDs");
        check_cuda(cudaMalloc((void**)&d.randoms, B*sizeof(float)), "allocate randoms");
        check_cuda(cudaMalloc((void**)&d.probabilities, bytes), "allocate probability scratch");
        check_cuda(cudaMalloc((void**)&d.cdf, bytes), "allocate CDF scratch");
        check_cuda(cudaMalloc((void**)&d.sampled, B*sizeof(std::int32_t)), "allocate sampled IDs");
        check_cuda(cudaMemcpyAsync(d.scores, scores.data(), bytes, cudaMemcpyHostToDevice, stream), "upload scores");
        check_cuda(cudaMemcpyAsync(d.ids, ids.data(), count*sizeof(std::int32_t), cudaMemcpyHostToDevice, stream), "upload IDs");
        check_cuda(cudaMemcpyAsync(d.randoms, randoms.data(), B*sizeof(float), cudaMemcpyHostToDevice, stream), "upload randoms");
        check_cuda(cudaStreamSynchronize(stream), "finish uploads");

        double p_errors[2], cdf_errors[2];
        for (int mode = 0; mode < 2; mode++) {
            launch_sampler(d, mode, temperature, stream);
            check_cuda(cudaMemcpyAsync(p.data(), d.probabilities, bytes, cudaMemcpyDeviceToHost, stream), "read probabilities");
            check_cuda(cudaMemcpyAsync(cdf.data(), d.cdf, bytes, cudaMemcpyDeviceToHost, stream), "read CDF");
            check_cuda(cudaMemcpyAsync(sampled.data(), d.sampled, B*sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream), "read sampled IDs");
            check_cuda(cudaStreamSynchronize(stream), "finish validation reads");
            if (!validate(p,cdf,sampled,ids,randoms,expected_p,expected_cdf,expected_ids,
                          B,k,p_errors[mode],cdf_errors[mode])) {
                std::fprintf(stderr, "Case %d, %s: no timing accepted for a failed check\n",
                    case_number, mode == 0 ? "serial" : "parallel");
                return 2;
            }
        }
        // check input buffers stayed fixed
        std::vector<float> scores_after(count), randoms_after(B);
        std::vector<std::int32_t> ids_after(count);
        check_cuda(cudaMemcpy(scores_after.data(),d.scores,bytes,cudaMemcpyDeviceToHost), "check scores unchanged");
        check_cuda(cudaMemcpy(ids_after.data(),d.ids,count*sizeof(std::int32_t),cudaMemcpyDeviceToHost), "check IDs unchanged");
        check_cuda(cudaMemcpy(randoms_after.data(),d.randoms,B*sizeof(float),cudaMemcpyDeviceToHost), "check randoms unchanged");
        if (std::memcmp(scores.data(),scores_after.data(),bytes) ||
            std::memcmp(ids.data(),ids_after.data(),count*sizeof(std::int32_t)) ||
            std::memcmp(randoms.data(),randoms_after.data(),B*sizeof(float))) {
            std::fprintf(stderr, "A sampler mutated an input buffer\n"); return 2;
        }

        // both methods pass before timing
        for (int i = 0; i < warmup; i++) {
            launch_sampler(d,0,temperature,stream);
            launch_sampler(d,1,temperature,stream);
        }
        check_cuda(cudaStreamSynchronize(stream), "finish warmup");
        std::vector<float> times[2];
        times[0].resize(repeats); times[1].resize(repeats);
        for (int r = 0; r < repeats; r++) {
            // alternate order to reduce bias
            for (int j = 0; j < 2; j++) {
                int mode = (r+j)%2;
                times[mode][r] = measure(d,mode,temperature,launches,stream,start,stop);
                if (!std::isfinite(times[mode][r]) || times[mode][r] <= 0) {
                    std::fprintf(stderr, "Invalid/zero event timing; try --launches-per-sample 100\n"); return 1;
                }
            }
        }
        std::cout << "CASE " << case_number << ' ' << B << ' ' << k << ' ' << temperature << '\n';
        for (int mode = 0; mode < 2; mode++) {
            const char* label = mode == 0 ? "serial" : "parallel";
            std::cout << "CHECK " << label << " PASS " << p_errors[mode] << ' ' << cdf_errors[mode] << '\n';
            std::cout << "TIMES " << label << ' ' << repeats;
            for (int r = 0; r < repeats; r++) std::cout << ' ' << times[mode][r];
            std::cout << '\n';
        }
        std::cout << "END\n";

        check_cuda(cudaFree(d.scores), "free scores");
        check_cuda(cudaFree(d.ids), "free IDs");
        check_cuda(cudaFree(d.randoms), "free randoms");
        check_cuda(cudaFree(d.probabilities), "free probabilities");
        check_cuda(cudaFree(d.cdf), "free CDF");
        check_cuda(cudaFree(d.sampled), "free sampled IDs");
        case_number++;
    }
    if (!std::cin.eof()) { std::fprintf(stderr, "Malformed input header\n"); return 1; }
    check_cuda(cudaEventDestroy(start), "destroy start event");
    check_cuda(cudaEventDestroy(stop), "destroy stop event");
    check_cuda(cudaStreamDestroy(stream), "destroy stream");
    return 0;
}
