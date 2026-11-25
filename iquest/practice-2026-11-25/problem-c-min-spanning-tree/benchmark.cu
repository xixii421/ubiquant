#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>

extern "C" void solution(const float* A, float* min_weight, size_t n);

// Exact CPU Prim's algorithm in FP64 (matches PyTorch reference_solution)
static float cpu_prim_mst(const std::vector<float>& A, int n) {
    if (n <= 1) return 0.0f;
    const float INF = std::numeric_limits<float>::infinity();
    std::vector<bool> in_mst(n, false);
    std::vector<float> min_e(n, INF);

    in_mst[0] = true;
    for (int j = 1; j < n; ++j) {
        float w = A[j];
        if (w > 0.0f) min_e[j] = w;
    }

    double total = 0.0;
    for (int step = 0; step < n - 1; ++step) {
        int best_u = -1;
        float best_w = INF;
        for (int i = 0; i < n; ++i) {
            if (!in_mst[i] && min_e[i] < best_w) {
                best_w = min_e[i];
                best_u = i;
            }
        }
        if (best_u == -1) return -1.0f;
        in_mst[best_u] = true;
        total += (double)best_w;
        const float* row = A.data() + (size_t)best_u * n;
        for (int j = 0; j < n; ++j) {
            if (!in_mst[j]) {
                float w = row[j];
                if (w > 0.0f && w < min_e[j]) {
                    min_e[j] = w;
                }
            }
        }
    }
    return (float)total;
}

static std::vector<float> make_random_graph(int n, int max_w, double density, uint32_t seed, bool force_connected = true) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> wdist(1, max_w);
    std::uniform_real_distribution<double> pdist(0.0, 1.0);
    std::vector<float> A((size_t)n * n, 0.0f);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (density >= 1.0 || pdist(rng) < density) {
                float w = (float)wdist(rng);
                A[(size_t)i * n + j] = w;
                A[(size_t)j * n + i] = w;
            }
        }
    }
    if (force_connected && n > 1) {
        for (int i = 1; i < n; ++i) {
            int p = rng() % i;
            if (A[(size_t)i * n + p] == 0.0f) {
                float w = (float)wdist(rng);
                A[(size_t)i * n + p] = w;
                A[(size_t)p * n + i] = w;
            }
        }
    }
    return A;
}

static std::vector<float> make_two_cliques_bridge(int n, int bridge_w, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> wdist(1, 100);
    std::vector<float> A((size_t)n * n, 0.0f);
    int half = n / 2;
    for (int i = 0; i < half; ++i) {
        for (int j = i + 1; j < half; ++j) {
            float w = (float)wdist(rng);
            A[(size_t)i * n + j] = A[(size_t)j * n + i] = w;
        }
    }
    for (int i = half; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            float w = (float)wdist(rng);
            A[(size_t)i * n + j] = A[(size_t)j * n + i] = w;
        }
    }
    if (bridge_w > 0 && half > 0 && half < n) {
        A[(size_t)(half - 1) * n + half] = (float)bridge_w;
        A[(size_t)half * n + (half - 1)] = (float)bridge_w;
    }
    return A;
}

static bool test_case(const std::string& name, const std::vector<float>& A, int n) {
    float ref = cpu_prim_mst(A, n);
    float* d_A = nullptr;
    float* d_out = nullptr;
    if (n > 0) cudaMalloc(&d_A, (size_t)n * n * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    if (n > 0) cudaMemcpy(d_A, A.data(), (size_t)n * n * sizeof(float), cudaMemcpyHostToDevice);
    float init_val = -999.0f;
    cudaMemcpy(d_out, &init_val, sizeof(float), cudaMemcpyHostToDevice);

    solution(d_A, d_out, (size_t)n);
    cudaError_t err = cudaDeviceSynchronize();
    float got = -999.0f;
    cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    if (d_A) cudaFree(d_A);
    cudaFree(d_out);

    bool ok = (err == cudaSuccess) && (std::fabs(got - ref) <= 1e-3f * std::max(1.0f, std::fabs(ref)));
    if (!ok) {
        printf("[FAIL] %-32s n=%-5d ref=%.1f got=%.1f err=%s\n",
               name.c_str(), n, ref, got, cudaGetErrorString(err));
    }
    return ok;
}

static float bench_case(const std::string& label, const std::vector<float>& A, int n, int warmup = 5, int iters = 20) {
    float ref = cpu_prim_mst(A, n);
    float* d_A = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_A, (size_t)n * n * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_A, A.data(), (size_t)n * n * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < warmup; ++i) {
        solution(d_A, d_out, (size_t)n);
    }
    cudaDeviceSynchronize();

    float got = 0.0f;
    cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = (std::fabs(got - ref) <= 1e-3f * std::max(1.0f, std::fabs(ref)));

    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    cudaEventRecord(st);
    for (int i = 0; i < iters; ++i) {
        solution(d_A, d_out, (size_t)n);
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, st, ed);
    ms /= iters;

    printf("  %-26s n=%-5d | %7.3f ms (%7.1f us) | ref=%10.1f got=%10.1f %s\n",
           label.c_str(), n, ms, ms * 1000.0f, ref, got, ok ? "OK" : "FAIL!");

    cudaEventDestroy(st);
    cudaEventDestroy(ed);
    cudaFree(d_A);
    cudaFree(d_out);
    return ms;
}

int main() {
    int passed = 0, total = 0;
    auto run_t = [&](const std::string& name, const std::vector<float>& A, int n) {
        total++;
        if (test_case(name, A, n)) passed++;
    };

    // Edge cases
    run_t("n=1", {0.0f}, 1);
    run_t("n=2 connected", {0, 5, 5, 0}, 2);
    run_t("n=2 disconnected", {0, 0, 0, 0}, 2);
    run_t("n=3 all equal", {0, 7, 7, 7, 0, 7, 7, 7, 0}, 3);
    run_t("n=7 random dense", make_random_graph(7, 10, 1.0, 1), 7);
    run_t("n=31 all-ones", make_random_graph(31, 1, 1.0, 2), 31);
    run_t("n=32 random", make_random_graph(32, 100, 1.0, 3), 32);
    run_t("n=33 sparse", make_random_graph(33, 1000, 0.1, 4), 33);
    run_t("n=64 two-cliques bridge", make_two_cliques_bridge(64, 99999, 5), 64);
    run_t("n=64 disconnected", make_two_cliques_bridge(64, 0, 6), 64);
    run_t("n=127 large weights", make_random_graph(127, 1000000, 1.0, 7), 127);
    run_t("n=257 sparse tree-ish", make_random_graph(257, 500, 0.02, 8), 257);
    run_t("n=513 all-equal", make_random_graph(513, 1, 1.0, 9), 513);
    run_t("n=512 two-cliques bridge", make_two_cliques_bridge(512, 1000000, 10), 512);
    run_t("n=1000 distinct weights", make_random_graph(1000, 10000000, 1.0, 11), 1000);
    run_t("n=1024 sparse 2%", make_random_graph(1024, 1000, 0.02, 12), 1024);

    // Path graph (adversarial high diameter)
    {
        int n = 500;
        std::vector<float> A((size_t)n * n, 0.0f);
        for (int i = 0; i < n - 1; ++i) {
            float w = (float)((i * 37) % 100 + 1);
            A[(size_t)i * n + (i + 1)] = A[(size_t)(i + 1) * n + i] = w;
        }
        run_t("n=500 path graph", A, n);
    }

    printf("Correctness suite: %d / %d passed\n\n", passed, total);
    if (passed != total) return 1;

    // Official sizes across 3 distributions
    const int sizes[] = {1024, 2048, 4096, 6144};

    printf("=== Dense uniform [1, 1000000] (distinct weights, hardest for Boruvka) ===\n");
    float sum1 = 0.0f;
    for (int n : sizes) {
        auto A = make_random_graph(n, 1000000, 1.0, 100 + n);
        sum1 += bench_case("dense-distinct", A, n);
    }
    printf("  TOTAL: %.3f ms (%.1f us)\n\n", sum1, sum1 * 1000.0f);

    printf("=== Dense uniform [1, 100] (many ties) ===\n");
    float sum2 = 0.0f;
    for (int n : sizes) {
        auto A = make_random_graph(n, 100, 1.0, 200 + n);
        sum2 += bench_case("dense-ties-100", A, n);
    }
    printf("  TOTAL: %.3f ms (%.1f us)\n\n", sum2, sum2 * 1000.0f);

    printf("=== Sparse 5%% density [1, 10000] ===\n");
    float sum3 = 0.0f;
    for (int n : sizes) {
        auto A = make_random_graph(n, 10000, 0.05, 300 + n);
        sum3 += bench_case("sparse-5pct", A, n);
    }
    printf("  TOTAL: %.3f ms (%.1f us)\n\n", sum3, sum3 * 1000.0f);

    printf("=== Adversarial: two cliques + 1 heavy bridge ===\n");
    float sum4 = 0.0f;
    for (int n : sizes) {
        auto A = make_two_cliques_bridge(n, 999999, 400 + n);
        sum4 += bench_case("two-cliques", A, n);
    }
    printf("  TOTAL: %.3f ms (%.1f us)\n", sum4, sum4 * 1000.0f);

    return 0;
}
