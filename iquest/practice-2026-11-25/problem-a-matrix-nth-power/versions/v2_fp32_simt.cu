#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace {

__device__ __forceinline__ void cp_async_ca_16(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void swizzle_cta(int bid, int tiles_m, int tiles_n, int group_m, int& tm, int& tn) {
    int num_in_grp = group_m * tiles_n;
    int grp_id = bid / num_in_grp;
    int first_m = grp_id * group_m;
    int grp_sz_m = min(tiles_m - first_m, group_m);
    tm = first_m + (bid % grp_sz_m);
    tn = (bid % num_in_grp) / grp_sz_m;
}

__device__ __forceinline__ void fma_8x8(float acc[8][8], const float rA[8], const float rB[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i][j] = fmaf(rA[i], rB[j], acc[i][j]);
}

__global__ void identity_kernel(float* __restrict__ out, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = size * size;
    if (idx < total) {
        size_t r = idx / size;
        size_t c = idx % size;
        out[idx] = (r == c) ? 1.0f : 0.0f;
    }
}

// Bounds-checked fallback SGEMM for arbitrary N (not multiple of 128)
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

// High-performance 64x128x16 Pure IEEE-754 FP32 SGEMM (128T, 4 blocks/SM on L4 & H20, 0 spills)
template <int GROUP_M>
__global__ void __launch_bounds__(128, 4)
sgemm_fp32_64x128x16(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    constexpr int BN = 128, BK = 16, SA_S = 68;
    __shared__ __align__(16) float sA[2][BK][SA_S];
    __shared__ __align__(16) float sB[2][BK][BN];

    int tid = threadIdx.x;
    int tile_m, tile_n;
    swizzle_cta(blockIdx.x, N >> 6, N >> 7, GROUP_M, tile_m, tile_n);
    int bm = tile_m << 6, bn = tile_n << 7;

    int warp_id = tid >> 5, lane_id = tid & 31;
    int warp_m = warp_id >> 1, warp_n = warp_id & 1;
    int lane_m = lane_id >> 3, lane_n = lane_id & 7;
    int m0 = (warp_m << 5) + (lane_m << 2), m1 = m0 + 16;
    int n0 = (warp_n << 6) + (lane_n << 2), n1 = n0 + 32;

    int a_m = tid >> 1, a_k = (tid & 1) << 2;
    int b_k = tid >> 5, b_n = (tid & 31) << 2;

    const size_t stride4 = (size_t)N << 2;
    const size_t stride8 = (size_t)N << 3;
    const size_t stride12 = stride8 + stride4;
    const size_t stride16 = (size_t)N << 4;

    const float* A_ptr = A + (size_t)(bm + a_m) * N + a_k;
    const float* B_ptr = B + (size_t)b_k * N + (bn + b_n);

    float acc[8][8] = {{0}};
    float rA[2][8], rB[2][8];

    {
        cp_async_ca_16(&sB[0][b_k + 0][b_n], B_ptr);
        cp_async_ca_16(&sB[0][b_k + 4][b_n], B_ptr + stride4);
        cp_async_ca_16(&sB[0][b_k + 8][b_n], B_ptr + stride8);
        cp_async_ca_16(&sB[0][b_k + 12][b_n], B_ptr + stride12);
        cp_async_commit();
        float4 a0 = *reinterpret_cast<const float4*>(A_ptr);
        float4 a1 = *reinterpret_cast<const float4*>(A_ptr + 8);
        sA[0][a_k + 0][a_m] = a0.x; sA[0][a_k + 1][a_m] = a0.y;
        sA[0][a_k + 2][a_m] = a0.z; sA[0][a_k + 3][a_m] = a0.w;
        sA[0][a_k + 8][a_m] = a1.x; sA[0][a_k + 9][a_m] = a1.y;
        sA[0][a_k + 10][a_m] = a1.z; sA[0][a_k + 11][a_m] = a1.w;
        cp_async_wait<0>();
    }
    __syncthreads();

    *reinterpret_cast<float4*>(&rA[0][0]) = *reinterpret_cast<const float4*>(&sA[0][0][m0]);
    *reinterpret_cast<float4*>(&rA[0][4]) = *reinterpret_cast<const float4*>(&sA[0][0][m1]);
    *reinterpret_cast<float4*>(&rB[0][0]) = *reinterpret_cast<const float4*>(&sB[0][0][n0]);
    *reinterpret_cast<float4*>(&rB[0][4]) = *reinterpret_cast<const float4*>(&sB[0][0][n1]);

    int cur = 0;
    for (int bk = BK; bk < N; bk += BK) {
        A_ptr += BK;
        B_ptr += stride16;
        int nxt = cur ^ 1;
        cp_async_ca_16(&sB[nxt][b_k + 0][b_n], B_ptr);
        cp_async_ca_16(&sB[nxt][b_k + 4][b_n], B_ptr + stride4);
        cp_async_ca_16(&sB[nxt][b_k + 8][b_n], B_ptr + stride8);
        cp_async_ca_16(&sB[nxt][b_k + 12][b_n], B_ptr + stride12);
        cp_async_commit();
        float4 a0 = *reinterpret_cast<const float4*>(A_ptr);
        float4 a1 = *reinterpret_cast<const float4*>(A_ptr + 8);

        #pragma unroll
        for (int k = 0; k < BK - 1; ++k) {
            int rc = k & 1, rn = rc ^ 1;
            *reinterpret_cast<float4*>(&rA[rn][0]) = *reinterpret_cast<const float4*>(&sA[cur][k + 1][m0]);
            *reinterpret_cast<float4*>(&rA[rn][4]) = *reinterpret_cast<const float4*>(&sA[cur][k + 1][m1]);
            *reinterpret_cast<float4*>(&rB[rn][0]) = *reinterpret_cast<const float4*>(&sB[cur][k + 1][n0]);
            *reinterpret_cast<float4*>(&rB[rn][4]) = *reinterpret_cast<const float4*>(&sB[cur][k + 1][n1]);
            fma_8x8(acc, rA[rc], rB[rc]);
        }

        sA[nxt][a_k + 0][a_m] = a0.x; sA[nxt][a_k + 1][a_m] = a0.y;
        sA[nxt][a_k + 2][a_m] = a0.z; sA[nxt][a_k + 3][a_m] = a0.w;
        sA[nxt][a_k + 8][a_m] = a1.x; sA[nxt][a_k + 9][a_m] = a1.y;
        sA[nxt][a_k + 10][a_m] = a1.z; sA[nxt][a_k + 11][a_m] = a1.w;
        cp_async_wait<0>();
        __syncthreads();

        *reinterpret_cast<float4*>(&rA[0][0]) = *reinterpret_cast<const float4*>(&sA[nxt][0][m0]);
        *reinterpret_cast<float4*>(&rA[0][4]) = *reinterpret_cast<const float4*>(&sA[nxt][0][m1]);
        *reinterpret_cast<float4*>(&rB[0][0]) = *reinterpret_cast<const float4*>(&sB[nxt][0][n0]);
        *reinterpret_cast<float4*>(&rB[0][4]) = *reinterpret_cast<const float4*>(&sB[nxt][0][n1]);
        fma_8x8(acc, rA[1], rB[1]);
        cur = nxt;
    }

    #pragma unroll
    for (int k = 0; k < BK - 1; ++k) {
        int rc = k & 1, rn = rc ^ 1;
        *reinterpret_cast<float4*>(&rA[rn][0]) = *reinterpret_cast<const float4*>(&sA[cur][k + 1][m0]);
        *reinterpret_cast<float4*>(&rA[rn][4]) = *reinterpret_cast<const float4*>(&sA[cur][k + 1][m1]);
        *reinterpret_cast<float4*>(&rB[rn][0]) = *reinterpret_cast<const float4*>(&sB[cur][k + 1][n0]);
        *reinterpret_cast<float4*>(&rB[rn][4]) = *reinterpret_cast<const float4*>(&sB[cur][k + 1][n1]);
        fma_8x8(acc, rA[rc], rB[rc]);
    }
    fma_8x8(acc, rA[1], rB[1]);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float* c0 = C + (size_t)(bm + m0 + i) * N + bn;
        float* c1 = C + (size_t)(bm + m1 + i) * N + bn;
        *reinterpret_cast<float4*>(c0 + n0) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(c0 + n1) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        *reinterpret_cast<float4*>(c1 + n0) = make_float4(acc[i + 4][0], acc[i + 4][1], acc[i + 4][2], acc[i + 4][3]);
        *reinterpret_cast<float4*>(c1 + n1) = make_float4(acc[i + 4][4], acc[i + 4][5], acc[i + 4][6], acc[i + 4][7]);
    }
}

// Split-K=4 64x128x16 Pure FP32 SGEMM for 512 & 1024 (128 / 512 blocks -> saturates H20 78 SMs!)
template <int KSPLIT>
__global__ void __launch_bounds__(128, 4)
sgemm_fp32_64x128x16_splitK(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C_part, int N) {
    constexpr int BN = 128, BK = 16, SA_S = 68;
    __shared__ __align__(16) float sA[2][BK][SA_S];
    __shared__ __align__(16) float sB[2][BK][BN];

    int tid = threadIdx.x;
    int tile_m, tile_n;
    swizzle_cta(blockIdx.x, N >> 6, N >> 7, 4, tile_m, tile_n);
    int ks = blockIdx.y;
    int bm = tile_m << 6, bn = tile_n << 7;

    int k_len = N / KSPLIT;
    int k_start = ks * k_len;
    int k_end = k_start + k_len;

    int warp_id = tid >> 5, lane_id = tid & 31;
    int warp_m = warp_id >> 1, warp_n = warp_id & 1;
    int lane_m = lane_id >> 3, lane_n = lane_id & 7;
    int m0 = (warp_m << 5) + (lane_m << 2), m1 = m0 + 16;
    int n0 = (warp_n << 6) + (lane_n << 2), n1 = n0 + 32;

    int a_m = tid >> 1, a_k = (tid & 1) << 2;
    int b_k = tid >> 5, b_n = (tid & 31) << 2;

    const size_t stride4 = (size_t)N << 2;
    const size_t stride8 = (size_t)N << 3;
    const size_t stride12 = stride8 + stride4;
    const size_t stride16 = (size_t)N << 4;

    const float* A_ptr = A + (size_t)(bm + a_m) * N + k_start + a_k;
    const float* B_ptr = B + (size_t)(k_start + b_k) * N + (bn + b_n);

    float acc[8][8] = {{0}};
    float rA[2][8], rB[2][8];

    {
        cp_async_ca_16(&sB[0][b_k + 0][b_n], B_ptr);
        cp_async_ca_16(&sB[0][b_k + 4][b_n], B_ptr + stride4);
        cp_async_ca_16(&sB[0][b_k + 8][b_n], B_ptr + stride8);
        cp_async_ca_16(&sB[0][b_k + 12][b_n], B_ptr + stride12);
        cp_async_commit();
        float4 a0 = *reinterpret_cast<const float4*>(A_ptr);
        float4 a1 = *reinterpret_cast<const float4*>(A_ptr + 8);
        sA[0][a_k + 0][a_m] = a0.x; sA[0][a_k + 1][a_m] = a0.y;
        sA[0][a_k + 2][a_m] = a0.z; sA[0][a_k + 3][a_m] = a0.w;
        sA[0][a_k + 8][a_m] = a1.x; sA[0][a_k + 9][a_m] = a1.y;
        sA[0][a_k + 10][a_m] = a1.z; sA[0][a_k + 11][a_m] = a1.w;
        cp_async_wait<0>();
    }
    __syncthreads();

    *reinterpret_cast<float4*>(&rA[0][0]) = *reinterpret_cast<const float4*>(&sA[0][0][m0]);
    *reinterpret_cast<float4*>(&rA[0][4]) = *reinterpret_cast<const float4*>(&sA[0][0][m1]);
    *reinterpret_cast<float4*>(&rB[0][0]) = *reinterpret_cast<const float4*>(&sB[0][0][n0]);
    *reinterpret_cast<float4*>(&rB[0][4]) = *reinterpret_cast<const float4*>(&sB[0][0][n1]);

    int cur = 0;
    for (int bk = k_start + BK; bk < k_end; bk += BK) {
        A_ptr += BK;
        B_ptr += stride16;
        int nxt = cur ^ 1;
        cp_async_ca_16(&sB[nxt][b_k + 0][b_n], B_ptr);
        cp_async_ca_16(&sB[nxt][b_k + 4][b_n], B_ptr + stride4);
        cp_async_ca_16(&sB[nxt][b_k + 8][b_n], B_ptr + stride8);
        cp_async_ca_16(&sB[nxt][b_k + 12][b_n], B_ptr + stride12);
        cp_async_commit();
        float4 a0 = *reinterpret_cast<const float4*>(A_ptr);
        float4 a1 = *reinterpret_cast<const float4*>(A_ptr + 8);

        #pragma unroll
        for (int k = 0; k < BK - 1; ++k) {
            int rc = k & 1, rn = rc ^ 1;
            *reinterpret_cast<float4*>(&rA[rn][0]) = *reinterpret_cast<const float4*>(&sA[cur][k + 1][m0]);
            *reinterpret_cast<float4*>(&rA[rn][4]) = *reinterpret_cast<const float4*>(&sA[cur][k + 1][m1]);
            *reinterpret_cast<float4*>(&rB[rn][0]) = *reinterpret_cast<const float4*>(&sB[cur][k + 1][n0]);
            *reinterpret_cast<float4*>(&rB[rn][4]) = *reinterpret_cast<const float4*>(&sB[cur][k + 1][n1]);
            fma_8x8(acc, rA[rc], rB[rc]);
        }

        sA[nxt][a_k + 0][a_m] = a0.x; sA[nxt][a_k + 1][a_m] = a0.y;
        sA[nxt][a_k + 2][a_m] = a0.z; sA[nxt][a_k + 3][a_m] = a0.w;
        sA[nxt][a_k + 8][a_m] = a1.x; sA[nxt][a_k + 9][a_m] = a1.y;
        sA[nxt][a_k + 10][a_m] = a1.z; sA[nxt][a_k + 11][a_m] = a1.w;
        cp_async_wait<0>();
        __syncthreads();

        *reinterpret_cast<float4*>(&rA[0][0]) = *reinterpret_cast<const float4*>(&sA[nxt][0][m0]);
        *reinterpret_cast<float4*>(&rA[0][4]) = *reinterpret_cast<const float4*>(&sA[nxt][0][m1]);
        *reinterpret_cast<float4*>(&rB[0][0]) = *reinterpret_cast<const float4*>(&sB[nxt][0][n0]);
        *reinterpret_cast<float4*>(&rB[0][4]) = *reinterpret_cast<const float4*>(&sB[nxt][0][n1]);
        fma_8x8(acc, rA[1], rB[1]);
        cur = nxt;
    }

    #pragma unroll
    for (int k = 0; k < BK - 1; ++k) {
        int rc = k & 1, rn = rc ^ 1;
        *reinterpret_cast<float4*>(&rA[rn][0]) = *reinterpret_cast<const float4*>(&sA[cur][k + 1][m0]);
        *reinterpret_cast<float4*>(&rA[rn][4]) = *reinterpret_cast<const float4*>(&sA[cur][k + 1][m1]);
        *reinterpret_cast<float4*>(&rB[rn][0]) = *reinterpret_cast<const float4*>(&sB[cur][k + 1][n0]);
        *reinterpret_cast<float4*>(&rB[rn][4]) = *reinterpret_cast<const float4*>(&sB[cur][k + 1][n1]);
        fma_8x8(acc, rA[rc], rB[rc]);
    }
    fma_8x8(acc, rA[1], rB[1]);

    float* C_out = C_part + (size_t)ks * N * N;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float* c0 = C_out + (size_t)(bm + m0 + i) * N + bn;
        float* c1 = C_out + (size_t)(bm + m1 + i) * N + bn;
        *reinterpret_cast<float4*>(c0 + n0) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(c0 + n1) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        *reinterpret_cast<float4*>(c1 + n0) = make_float4(acc[i + 4][0], acc[i + 4][1], acc[i + 4][2], acc[i + 4][3]);
        *reinterpret_cast<float4*>(c1 + n1) = make_float4(acc[i + 4][4], acc[i + 4][5], acc[i + 4][6], acc[i + 4][7]);
    }
}

template <int KSPLIT>
__global__ void reduce_splitK(const float* __restrict__ C_part, float* __restrict__ C, size_t elems4) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elems4) return;
    const float4* src = reinterpret_cast<const float4*>(C_part);
    float4 sum = src[idx];
    #pragma unroll
    for (int s = 1; s < KSPLIT; ++s) {
        float4 v = src[idx + (size_t)s * elems4];
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
    reinterpret_cast<float4*>(C)[idx] = sum;
}

struct Workspace {
    float* buf[3] = {nullptr, nullptr, nullptr};
    size_t cap[3] = {0, 0, 0};

    float* get(int idx, size_t elems) {
        if (elems > cap[idx]) {
            if (buf[idx]) cudaFree(buf[idx]);
            cudaMalloc(&buf[idx], elems * sizeof(float));
            cap[idx] = elems;
        }
        return buf[idx];
    }

    ~Workspace() {
        for (int i = 0; i < 3; ++i) {
            if (buf[i]) cudaFree(buf[i]);
        }
    }
};

static Workspace g_ws;

static void launch_gemm(const float* A, const float* B, float* C, size_t size) {
    if ((size & 127) == 0 && size >= 128) {
        int N = (int)size;
        unsigned blocks_2d = (unsigned)((size >> 6) * (size >> 7));
        if (size <= 1024 && (size & 63) == 0) {
            // Split-K=4 saturates all 78 SMs on H20 and 58 SMs on L4 for N=512 & N=1024
            float* d_part = g_ws.get(2, 4 * size * size);
            dim3 g(blocks_2d, 4);
            sgemm_fp32_64x128x16_splitK<4><<<g, 128>>>(A, B, d_part, N);
            size_t e4 = (size * size) >> 2;
            reduce_splitK<4><<<(unsigned)((e4 + 255) / 256), 256>>>(d_part, C, e4);
        } else {
            sgemm_fp32_64x128x16<4><<<blocks_2d, 128>>>(A, B, C, N);
        }
    } else {
        dim3 block(32, 32);
        dim3 grid((unsigned)((size + 31) / 32), (unsigned)((size + 31) / 32));
        sgemm_general_32<<<grid, block>>>(A, B, C, (int)size);
    }
}

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

    // Power-of-2 fast path: 0 temp buffers for n=2, 1 temp buffer for n=4,8,16,...
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

    // General binary exponentiation (no wasted multiply by identity)
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
