#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace {

__global__ void identity_kernel(float* __restrict__ out, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = size * size;
    if (idx < total) {
        size_t r = idx / size;
        size_t c = idx % size;
        out[idx] = (r == c) ? 1.0f : 0.0f;
    }
}

// General fallback for arbitrary N not divisible by 128
__global__ void sgemm_general_32(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int N) {
    constexpr int T = 32;
    __shared__ float sA[T][T + 1];
    __shared__ float sB[T][T + 1];

    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < N; t += T) {
        int aCol = t + threadIdx.x;
        int bRow = t + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[(size_t)row * N + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[(size_t)bRow * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; ++k) {
            acc = fmaf(sA[threadIdx.y][k], sB[k][threadIdx.x], acc);
        }
        __syncthreads();
    }
    if (row < N && col < N) {
        C[(size_t)row * N + col] = acc;
    }
}

// High-performance 128x128x8 double-buffered FP32 SGEMM for N % 128 == 0
__global__ void __launch_bounds__(256, 2)
sgemm_128x128x8_dbuf(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int N) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int SA_STRIDE = BM + 4; // 132: 16B aligned & bank-conflict-free transpose
    constexpr int GROUP_M = 8;

    __shared__ __align__(16) float sA[2][BK][SA_STRIDE];
    __shared__ __align__(16) float sB[2][BK][BN];

    const int tid = threadIdx.x;
    const int num_tiles_m = N >> 7;
    const int num_tiles_n = N >> 7;

    // L2-friendly CTA swizzling
    const int bid = blockIdx.x;
    const int num_bid_in_group = GROUP_M * num_tiles_n;
    const int group_id = bid / num_bid_in_group;
    const int first_tile_m = group_id * GROUP_M;
    const int group_size_m = min(num_tiles_m - first_tile_m, GROUP_M);
    const int tile_m = first_tile_m + (bid % group_size_m);
    const int tile_n = (bid % num_bid_in_group) / group_size_m;

    const int bm = tile_m << 7;
    const int bn = tile_n << 7;

    // Warp and lane mapping: 4x2 warps -> each warp computes 32x64
    // Within warp: 4x8 lanes -> each lane computes 8x8 via 2x2 blocks of 4x4
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_m = warp_id >> 1; // 0..3
    const int warp_n = warp_id & 1;  // 0..1
    const int lane_m = lane_id >> 3; // 0..3
    const int lane_n = lane_id & 7;  // 0..7

    const int m0 = (warp_m << 5) + (lane_m << 2);
    const int m1 = m0 + 16;
    const int n0 = (warp_n << 6) + (lane_n << 2);
    const int n1 = n0 + 32;

    // Global load coordinates for A (128x8 = 256 float4) and B (8x128 = 256 float4)
    const int a_load_m = tid >> 1;        // 0..127
    const int a_load_k = (tid & 1) << 2;  // 0 or 4
    const int b_load_k = tid >> 5;        // 0..7
    const int b_load_n = (tid & 31) << 2; // 0, 4, ..., 124

    const float* A_ptr = A + (size_t)(bm + a_load_m) * N + a_load_k;
    const float* B_ptr = B + (size_t)b_load_k * N + (bn + b_load_n);

    // Accumulators: 8x8
    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i][j] = 0.0f;

    // Double-buffered register fragments
    float rA[2][8];
    float rB[2][8];

    // Prologue: load tile bk = 0 into smem stage 0
    {
        float4 ldgA = *reinterpret_cast<const float4*>(A_ptr);
        float4 ldgB = *reinterpret_cast<const float4*>(B_ptr);
        sA[0][a_load_k + 0][a_load_m] = ldgA.x;
        sA[0][a_load_k + 1][a_load_m] = ldgA.y;
        sA[0][a_load_k + 2][a_load_m] = ldgA.z;
        sA[0][a_load_k + 3][a_load_m] = ldgA.w;
        *reinterpret_cast<float4*>(&sB[0][b_load_k][b_load_n]) = ldgB;
    }
    __syncthreads();

    // Load k = 0 from stage 0 into register buffer 0
    *reinterpret_cast<float4*>(&rA[0][0]) = *reinterpret_cast<const float4*>(&sA[0][0][m0]);
    *reinterpret_cast<float4*>(&rA[0][4]) = *reinterpret_cast<const float4*>(&sA[0][0][m1]);
    *reinterpret_cast<float4*>(&rB[0][0]) = *reinterpret_cast<const float4*>(&sB[0][0][n0]);
    *reinterpret_cast<float4*>(&rB[0][4]) = *reinterpret_cast<const float4*>(&sB[0][0][n1]);

    int cur = 0;
    for (int bk = BK; bk < N; bk += BK) {
        A_ptr += BK;
        B_ptr += (size_t)BK * N;

        // Prefetch next global tile into registers early
        float4 ldgA = *reinterpret_cast<const float4*>(A_ptr);
        float4 ldgB = *reinterpret_cast<const float4*>(B_ptr);

        #pragma unroll
        for (int k = 0; k < BK - 1; ++k) {
            int r_cur = k & 1;
            int r_nxt = r_cur ^ 1;
            *reinterpret_cast<float4*>(&rA[r_nxt][0]) = *reinterpret_cast<const float4*>(&sA[cur][k + 1][m0]);
            *reinterpret_cast<float4*>(&rA[r_nxt][4]) = *reinterpret_cast<const float4*>(&sA[cur][k + 1][m1]);
            *reinterpret_cast<float4*>(&rB[r_nxt][0]) = *reinterpret_cast<const float4*>(&sB[cur][k + 1][n0]);
            *reinterpret_cast<float4*>(&rB[r_nxt][4]) = *reinterpret_cast<const float4*>(&sB[cur][k + 1][n1]);

            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[i][j] = fmaf(rA[r_cur][i], rB[r_cur][j], acc[i][j]);
        }

        // Write prefetched global tile to next smem stage
        int nxt = cur ^ 1;
        sA[nxt][a_load_k + 0][a_load_m] = ldgA.x;
        sA[nxt][a_load_k + 1][a_load_m] = ldgA.y;
        sA[nxt][a_load_k + 2][a_load_m] = ldgA.z;
        sA[nxt][a_load_k + 3][a_load_m] = ldgA.w;
        *reinterpret_cast<float4*>(&sB[nxt][b_load_k][b_load_n]) = ldgB;

        __syncthreads();

        // Prefetch k = 0 of nxt stage into rA[0]/rB[0] while rA[1]/rB[1] still holds k = 7 of cur
        *reinterpret_cast<float4*>(&rA[0][0]) = *reinterpret_cast<const float4*>(&sA[nxt][0][m0]);
        *reinterpret_cast<float4*>(&rA[0][4]) = *reinterpret_cast<const float4*>(&sA[nxt][0][m1]);
        *reinterpret_cast<float4*>(&rB[0][0]) = *reinterpret_cast<const float4*>(&sB[nxt][0][n0]);
        *reinterpret_cast<float4*>(&rB[0][4]) = *reinterpret_cast<const float4*>(&sB[nxt][0][n1]);

        // Compute k = 7 (r_cur = 1)
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[i][j] = fmaf(rA[1][i], rB[1][j], acc[i][j]);

        cur = nxt;
    }

    // Epilogue: last K tile
    #pragma unroll
    for (int k = 0; k < BK - 1; ++k) {
        int r_cur = k & 1;
        int r_nxt = r_cur ^ 1;
        *reinterpret_cast<float4*>(&rA[r_nxt][0]) = *reinterpret_cast<const float4*>(&sA[cur][k + 1][m0]);
        *reinterpret_cast<float4*>(&rA[r_nxt][4]) = *reinterpret_cast<const float4*>(&sA[cur][k + 1][m1]);
        *reinterpret_cast<float4*>(&rB[r_nxt][0]) = *reinterpret_cast<const float4*>(&sB[cur][k + 1][n0]);
        *reinterpret_cast<float4*>(&rB[r_nxt][4]) = *reinterpret_cast<const float4*>(&sB[cur][k + 1][n1]);

        #pragma unroll
        for (int i = 0; i < 8; ++i)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[i][j] = fmaf(rA[r_cur][i], rB[r_cur][j], acc[i][j]);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i][j] = fmaf(rA[1][i], rB[1][j], acc[i][j]);

    // Vectorized float4 stores to C
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float* c_row0 = C + (size_t)(bm + m0 + i) * N + bn;
        float* c_row1 = C + (size_t)(bm + m1 + i) * N + bn;
        *reinterpret_cast<float4*>(c_row0 + n0) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(c_row0 + n1) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        *reinterpret_cast<float4*>(c_row1 + n0) = make_float4(acc[i + 4][0], acc[i + 4][1], acc[i + 4][2], acc[i + 4][3]);
        *reinterpret_cast<float4*>(c_row1 + n1) = make_float4(acc[i + 4][4], acc[i + 4][5], acc[i + 4][6], acc[i + 4][7]);
    }
}

static void launch_gemm(const float* A, const float* B, float* C, size_t size) {
    if ((size & 127) == 0 && size >= 128) {
        unsigned tiles = (unsigned)(size >> 7);
        sgemm_128x128x8_dbuf<<<tiles * tiles, 256>>>(A, B, C, (int)size);
    } else {
        dim3 block(32, 32);
        dim3 grid((unsigned)((size + 31) / 32), (unsigned)((size + 31) / 32));
        sgemm_general_32<<<grid, block>>>(A, B, C, (int)size);
    }
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
