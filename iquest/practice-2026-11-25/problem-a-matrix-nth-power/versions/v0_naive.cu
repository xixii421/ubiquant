#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace {

constexpr int TILE = 32;

__global__ void identity_kernel(float* __restrict__ out, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = size * size;
    if (idx < total) {
        size_t r = idx / size;
        size_t c = idx % size;
        out[idx] = (r == c) ? 1.0f : 0.0f;
    }
}

__global__ void sgemm_naive_tiled(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int N) {
    __shared__ float sA[TILE][TILE + 1];
    __shared__ float sB[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < N; t += TILE) {
        int aCol = t + threadIdx.x;
        int bRow = t + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[(size_t)row * N + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[(size_t)bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc = fmaf(sA[threadIdx.y][k], sB[k][threadIdx.x], acc);
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[(size_t)row * N + col] = acc;
    }
}

static void launch_gemm(const float* A, const float* B, float* C, size_t size) {
    dim3 block(TILE, TILE);
    dim3 grid((unsigned)((size + TILE - 1) / TILE), (unsigned)((size + TILE - 1) / TILE));
    sgemm_naive_tiled<<<grid, block>>>(A, B, C, (int)size);
}

struct Workspace {
    float* buf[2] = {nullptr, nullptr};
    size_t cap[2] = {0, 0};

    float* get(int idx, size_t elems) {
        if (elems > cap[idx]) {
            if (buf[idx]) cudaFree(buf[idx]);
            cudaMalloc(&buf[idx], elems * sizeof(float));
            cap[idx] = elems;
        }
        return buf[idx];
    }

    ~Workspace() {
        for (int i = 0; i < 2; ++i) {
            if (buf[i]) cudaFree(buf[i]);
        }
    }
};

static Workspace g_ws;

} // namespace

// Note: input_matrix, output_matrix are device pointers
extern "C" void solution(const float* input_matrix, const size_t n, float* output_matrix, size_t size) {
    if (size == 0) return;
    size_t elems = size * size;
    size_t bytes = elems * sizeof(float);

    if (n == 0) {
        int threads = 256;
        int blocks = (int)((elems + threads - 1) / threads);
        identity_kernel<<<blocks, threads>>>(output_matrix, size);
        return;
    }
    if (n == 1) {
        cudaMemcpyAsync(output_matrix, input_matrix, bytes, cudaMemcpyDeviceToDevice);
        return;
    }

    // Power of 2 fast path: 0 temp buffers for n=2, 1 temp buffer for n=4,8,16,...
    if ((n & (n - 1)) == 0) {
        int steps = __builtin_ctzll(n);
        if (steps == 1) {
            launch_gemm(input_matrix, input_matrix, output_matrix, size);
            return;
        }
        float* tmp0 = g_ws.get(0, elems);
        float* dst = (steps & 1) ? output_matrix : tmp0;
        float* alt = (steps & 1) ? tmp0 : output_matrix;
        launch_gemm(input_matrix, input_matrix, dst, size);
        for (int s = 1; s < steps; ++s) {
            launch_gemm(dst, dst, alt, size);
            std::swap(dst, alt);
        }
        return;
    }

    // General binary exponentiation
    float* tmp0 = g_ws.get(0, elems);
    float* tmp1 = g_ws.get(1, elems);
    bool has_res = false;
    const float* base = input_matrix;
    float* res = output_matrix;
    float* scratch = tmp0;

    size_t p = n;
    while (p > 0) {
        if (p & 1) {
            if (!has_res) {
                cudaMemcpyAsync(res, base, bytes, cudaMemcpyDeviceToDevice);
                has_res = true;
            } else {
                launch_gemm(res, base, scratch, size);
                std::swap(res, scratch);
            }
        }
        p >>= 1;
        if (p > 0) {
            float* next_base = nullptr;
            if (tmp1 != res && tmp1 != base) next_base = tmp1;
            else if (tmp0 != res && tmp0 != base) next_base = tmp0;
            else next_base = output_matrix;
            launch_gemm(base, base, next_base, size);
            base = next_base;
            if (output_matrix != res && output_matrix != base) scratch = output_matrix;
            else if (tmp0 != res && tmp0 != base) scratch = tmp0;
            else scratch = tmp1;
        }
    }
    if (res != output_matrix) {
        cudaMemcpyAsync(output_matrix, res, bytes, cudaMemcpyDeviceToDevice);
    }
}
