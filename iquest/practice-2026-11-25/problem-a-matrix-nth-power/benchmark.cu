#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>

// Extern declaration matching the IQuest Competition signature
extern "C" void solution(const float* input_matrix, const size_t n, float* output_matrix, size_t size);

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t st = (call);                                            \
        if (st != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                    (int)st);                                                  \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

__global__ void ref_identity_kernel(float* out, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = size * size;
    if (idx < total) {
        size_t r = idx / size;
        size_t c = idx % size;
        out[idx] = (r == c) ? 1.0f : 0.0f;
    }
}

// Reference cuBLAS matrix power (row-major C = A * B <=> col-major C^T = B^T * A^T)
static void cublas_matpow(cublasHandle_t handle, const float* d_in, size_t n,
                          float* d_out, size_t size, float* d_tmp0, float* d_tmp1) {
    size_t bytes = size * size * sizeof(float);
    if (n == 0) {
        int threads = 256;
        int blocks = (int)((size * size + threads - 1) / threads);
        ref_identity_kernel<<<blocks, threads>>>(d_out, size);
        return;
    }
    if (n == 1) {
        CUDA_CHECK(cudaMemcpyAsync(d_out, d_in, bytes, cudaMemcpyDeviceToDevice));
        return;
    }

    auto gemm = [&](const float* A, const float* B, float* C) {
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 (int)size, (int)size, (int)size,
                                 &alpha, B, (int)size, A, (int)size,
                                 &beta, C, (int)size));
    };

    // Power-of-2 fast path (1 temp buffer, 0 extra copies)
    if ((n & (n - 1)) == 0) {
        int steps = __builtin_ctzll(n);
        if (steps == 1) {
            gemm(d_in, d_in, d_out);
            return;
        }
        float* dst = (steps & 1) ? d_out : d_tmp0;
        float* alt = (steps & 1) ? d_tmp0 : d_out;
        gemm(d_in, d_in, dst);
        for (int s = 1; s < steps; ++s) {
            gemm(dst, dst, alt);
            std::swap(dst, alt);
        }
        return;
    }

    // General binary exponentiation
    bool has_res = false;
    const float* base = d_in;
    float* res = d_out;
    float* scratch = d_tmp0;
    float* base_buf = d_tmp1;

    size_t p = n;
    while (p > 0) {
        if (p & 1) {
            if (!has_res) {
                CUDA_CHECK(cudaMemcpyAsync(res, base, bytes, cudaMemcpyDeviceToDevice));
                has_res = true;
            } else {
                gemm(res, base, scratch);
                std::swap(res, scratch);
            }
        }
        p >>= 1;
        if (p > 0) {
            float* next_base = (base == base_buf) ? ((res == d_out) ? d_tmp0 : d_out) : base_buf;
            // Ensure next_base != res and != base
            if (next_base == res) {
                if (d_out != res && d_out != base) next_base = d_out;
                else if (d_tmp0 != res && d_tmp0 != base) next_base = d_tmp0;
                else next_base = d_tmp1;
            }
            gemm(base, base, next_base);
            base = next_base;
            // Keep scratch disjoint from res and base
            if (d_out != res && d_out != base) scratch = d_out;
            else if (d_tmp0 != res && d_tmp0 != base) scratch = d_tmp0;
            else scratch = d_tmp1;
        }
    }
    if (res != d_out) {
        CUDA_CHECK(cudaMemcpyAsync(d_out, res, bytes, cudaMemcpyDeviceToDevice));
    }
}

static void init_matrix(std::vector<float>& h, size_t size, uint32_t seed) {
    std::mt19937 rng(seed);
    // Scale by 1/sqrt(size) so A^8 stays O(1) without overflow/underflow
    float scale = 1.0f / std::sqrt((float)size);
    std::uniform_real_distribution<float> dist(-scale, scale);
    h.resize(size * size);
    for (size_t i = 0; i < size * size; ++i) {
        h[i] = dist(rng);
    }
}

struct ErrorStats {
    double max_abs;
    double max_rel;
    double l2_rel;
    bool has_nan;
};

static ErrorStats compare_matrices(const std::vector<float>& ref, const std::vector<float>& got) {
    double max_abs = 0.0, max_rel = 0.0;
    double diff_sq = 0.0, ref_sq = 0.0;
    bool has_nan = false;
    for (size_t i = 0; i < ref.size(); ++i) {
        double r = ref[i], g = got[i];
        if (std::isnan(g) || std::isinf(g)) has_nan = true;
        double d = std::abs(r - g);
        if (d > max_abs) max_abs = d;
        double denom = std::max(std::abs(r), 1e-6);
        double rel = d / denom;
        if (rel > max_rel) max_rel = rel;
        diff_sq += d * d;
        ref_sq += r * r;
    }
    double l2_rel = std::sqrt(diff_sq) / std::max(std::sqrt(ref_sq), 1e-12);
    return {max_abs, max_rel, l2_rel, has_nan};
}

int main(int argc, char** argv) {
    bool skip_edge = (argc > 1 && strcmp(argv[1], "--bench-only") == 0);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs, L2 %.1f MB)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.l2CacheSize / (1024.0 * 1024.0));

    cublasHandle_t cublas_fp32, cublas_tf32;
    CUBLAS_CHECK(cublasCreate(&cublas_fp32));
    CUBLAS_CHECK(cublasSetMathMode(cublas_fp32, CUBLAS_PEDANTIC_MATH));
    CUBLAS_CHECK(cublasCreate(&cublas_tf32));
    CUBLAS_CHECK(cublasSetMathMode(cublas_tf32, CUBLAS_TF32_TENSOR_OP_MATH));

    const size_t max_elems = 2048ULL * 2048ULL;
    float *d_in, *d_out, *d_ref, *d_tmp0, *d_tmp1;
    CUDA_CHECK(cudaMalloc(&d_in, max_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, max_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, max_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp0, max_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp1, max_elems * sizeof(float)));

    // 1. Edge-case correctness tests
    if (!skip_edge) {
        printf("\n=== Edge-Case Correctness Tests ===\n");
        struct EdgeCase { size_t size; size_t n; };
        std::vector<EdgeCase> edges = {
            {1, 0}, {1, 1}, {1, 2}, {1, 5},
            {7, 0}, {7, 1}, {7, 2}, {7, 3}, {7, 7},
            {31, 3}, {32, 4}, {33, 5},
            {63, 2}, {64, 0}, {64, 1}, {64, 3}, {64, 6}, {65, 4},
            {127, 2}, {128, 3}, {128, 5}, {129, 3},
            {255, 2}, {256, 7}, {513, 2}, {512, 0}, {512, 1}, {512, 3}, {512, 5}
        };
        bool all_pass = true;
        for (auto tc : edges) {
            std::vector<float> h_in, h_ref(tc.size * tc.size), h_got(tc.size * tc.size);
            init_matrix(h_in, tc.size, (uint32_t)(tc.size * 131 + tc.n));
            CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), tc.size * tc.size * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_out, 0x7f, tc.size * tc.size * sizeof(float)));

            cublas_matpow(cublas_fp32, d_in, tc.n, d_ref, tc.size, d_tmp0, d_tmp1);
            solution(d_in, tc.n, d_out, tc.size);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, tc.size * tc.size * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_got.data(), d_out, tc.size * tc.size * sizeof(float), cudaMemcpyDeviceToHost));
            auto st = compare_matrices(h_ref, h_got);
            bool ok = !st.has_nan && (st.max_abs < 1e-3 || st.l2_rel < 1e-4);
            if (!ok) {
                printf("FAIL size=%4zu n=%2zu | max_abs=%.3e max_rel=%.3e l2_rel=%.3e\n",
                       tc.size, tc.n, st.max_abs, st.max_rel, st.l2_rel);
                all_pass = false;
            }
        }
        printf("Edge cases: %s (%zu tests)\n", all_pass ? "ALL PASSED" : "FAILED", edges.size());
        if (!all_pass) return 1;
    }

    // 2. Official 9 test cases: correctness + timing vs cuBLAS FP32 & cuBLAS TF32
    struct BenchCase { size_t size; size_t n; };
    std::vector<BenchCase> cases = {
        {512, 2}, {512, 4}, {512, 8},
        {1024, 2}, {1024, 4}, {1024, 8},
        {2048, 2}, {2048, 4}, {2048, 8}
    };

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    printf("\n=== Official Benchmark Suite (9 cases) ===\n");
    printf("%-14s | %9s %7s | %9s %7s | %9s %7s | %8s | %9s %9s\n",
           "Case", "Ours(ms)", "TFLOPS", "cuFP32", "TFLOPS", "cuTF32", "TFLOPS",
           "vsFP32", "max_abs", "l2_rel");
    printf("-----------------------------------------------------------------------------------------------------------------\n");

    double sum_ours = 0.0, sum_fp32 = 0.0, sum_tf32 = 0.0, sum_flops = 0.0;

    for (auto tc : cases) {
        std::vector<float> h_in, h_ref(tc.size * tc.size), h_got(tc.size * tc.size);
        init_matrix(h_in, tc.size, (uint32_t)(tc.size * 17 + tc.n));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), tc.size * tc.size * sizeof(float), cudaMemcpyHostToDevice));

        // Verify correctness against cuBLAS FP32
        cublas_matpow(cublas_fp32, d_in, tc.n, d_ref, tc.size, d_tmp0, d_tmp1);
        solution(d_in, tc.n, d_out, tc.size);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, tc.size * tc.size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_got.data(), d_out, tc.size * tc.size * sizeof(float), cudaMemcpyDeviceToHost));
        auto st = compare_matrices(h_ref, h_got);

        // Also check cuBLAS TF32 accuracy once for reference
        int gemms = __builtin_ctzll(tc.n);
        double flops = (double)gemms * 2.0 * (double)tc.size * (double)tc.size * (double)tc.size;
        sum_flops += flops;

        const int warmup = 5;
        const int iters = (tc.size <= 1024) ? 30 : 15;

        auto bench_fn = [&](auto fn) -> double {
            for (int i = 0; i < warmup; ++i) fn();
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> times(iters);
            for (int i = 0; i < iters; ++i) {
                CUDA_CHECK(cudaEventRecord(ev_start));
                fn();
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                CUDA_CHECK(cudaEventElapsedTime(&times[i], ev_start, ev_stop));
            }
            std::sort(times.begin(), times.end());
            return times[iters / 2]; // median ms
        };

        double ms_ours = bench_fn([&]() { solution(d_in, tc.n, d_out, tc.size); });
        double ms_fp32 = bench_fn([&]() { cublas_matpow(cublas_fp32, d_in, tc.n, d_ref, tc.size, d_tmp0, d_tmp1); });
        double ms_tf32 = bench_fn([&]() { cublas_matpow(cublas_tf32, d_in, tc.n, d_ref, tc.size, d_tmp0, d_tmp1); });

        sum_ours += ms_ours;
        sum_fp32 += ms_fp32;
        sum_tf32 += ms_tf32;

        double tf_ours = (flops / (ms_ours * 1e-3)) * 1e-12;
        double tf_fp32 = (flops / (ms_fp32 * 1e-3)) * 1e-12;
        double tf_tf32 = (flops / (ms_tf32 * 1e-3)) * 1e-12;

        char label[32];
        snprintf(label, sizeof(label), "%zux%zu p=%zu", tc.size, tc.size, tc.n);
        printf("%-14s | %9.4f %7.2f | %9.4f %7.2f | %9.4f %7.2f | %7.2fx | %.2e %.2e%s\n",
               label, ms_ours, tf_ours, ms_fp32, tf_fp32, ms_tf32, tf_tf32,
               ms_fp32 / ms_ours, st.max_abs, st.l2_rel,
               (st.has_nan || st.l2_rel > 1e-3) ? " [BAD]" : "");
    }
    printf("-----------------------------------------------------------------------------------------------------------------\n");
    printf("%-14s | %9.4f %7.2f | %9.4f %7.2f | %9.4f %7.2f | %7.2fx |\n",
           "TOTAL", sum_ours, (sum_flops / (sum_ours * 1e-3)) * 1e-12,
           sum_fp32, (sum_flops / (sum_fp32 * 1e-3)) * 1e-12,
           sum_tf32, (sum_flops / (sum_tf32 * 1e-3)) * 1e-12,
           sum_fp32 / sum_ours);

    // Report TF32 error on 2048x2048 p=8 for architectural insight
    {
        size_t sz = 2048, n = 8;
        std::vector<float> h_in, h_fp32(sz*sz), h_tf32(sz*sz);
        init_matrix(h_in, sz, 2048*17 + 8);
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), sz*sz*sizeof(float), cudaMemcpyHostToDevice));
        cublas_matpow(cublas_fp32, d_in, n, d_ref, sz, d_tmp0, d_tmp1);
        cublas_matpow(cublas_tf32, d_in, n, d_out, sz, d_tmp0, d_tmp1);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_fp32.data(), d_ref, sz*sz*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tf32.data(), d_out, sz*sz*sizeof(float), cudaMemcpyDeviceToHost));
        auto st = compare_matrices(h_fp32, h_tf32);
        printf("\n[Info] cuBLAS TF32 error vs FP32 on 2048x2048 p=8: max_abs=%.3e max_rel=%.3e l2_rel=%.3e\n",
               st.max_abs, st.max_rel, st.l2_rel);
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_tmp0));
    CUDA_CHECK(cudaFree(d_tmp1));
    CUBLAS_CHECK(cublasDestroy(cublas_fp32));
    CUBLAS_CHECK(cublasDestroy(cublas_tf32));
    return 0;
}
