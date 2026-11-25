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

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ uint32_t f32_to_tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void mma_m16n8k8_tf32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
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

// ============================================================================
// Pure-PTX TF32 Tensor Core GEMM (64x128x16, 128T) with full cp.async.ca
// 0 spills, 4 blocks/SM on H20, bank-conflict-free ldmatrix + LDS.32
// ============================================================================
template <int GROUP_M, int KSPLIT>
__global__ void __launch_bounds__(128, 4)
sgemm_tf32_64x128x16(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    constexpr int BM = 64, BN = 128, BK = 16;
    constexpr int SA_S = BK + 4; // 20 floats = 80B (16B aligned, bank-conflict-free ldmatrix)
    constexpr int SB_S = BN + 8; // 136 floats = 544B (16B aligned, bank-conflict-free LDS.32)

    __shared__ __align__(16) float sA[2][BM][SA_S];
    __shared__ __align__(16) float sB[2][BK][SB_S];

    int tid = threadIdx.x;
    int tile_m, tile_n;
    swizzle_cta(blockIdx.x, N >> 6, N >> 7, GROUP_M, tile_m, tile_n);
    int bm = tile_m << 6, bn = tile_n << 7;

    int ks = (KSPLIT > 1) ? blockIdx.y : 0;
    int k_len = (KSPLIT > 1) ? (N / KSPLIT) : N;
    int k_start = ks * k_len;
    int k_end = k_start + k_len;

    int warp_id = tid >> 5, lane_id = tid & 31;
    int warp_m = warp_id >> 1, warp_n = warp_id & 1;

    int a_m = tid >> 2, a_k = (tid & 3) << 2; // a_m: 0..31 (+32)
    int b_k = tid >> 5, b_n = (tid & 31) << 2; // b_k: 0..3 (+4, +8, +12)

    const size_t stride4 = (size_t)N << 2;
    const size_t stride8 = (size_t)N << 3;
    const size_t stride12 = stride8 + stride4;
    const size_t stride16 = (size_t)N << 4;

    const float* A_ptr0 = A + (size_t)(bm + a_m) * N + k_start + a_k;
    const float* A_ptr1 = A + (size_t)(bm + a_m + 32) * N + k_start + a_k;
    const float* B_ptr = B + (size_t)(k_start + b_k) * N + (bn + b_n);

    float acc[2][8][4];
    #pragma unroll
    for (int m = 0; m < 2; ++m)
        #pragma unroll
        for (int n = 0; n < 8; ++n)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[m][n][r] = 0.0f;

    auto issue_stage = [&](int st, const float* Ap0, const float* Ap1, const float* Bp) {
        cp_async_ca_16(&sA[st][a_m][a_k], Ap0);
        cp_async_ca_16(&sA[st][a_m + 32][a_k], Ap1);
        cp_async_ca_16(&sB[st][b_k + 0][b_n], Bp);
        cp_async_ca_16(&sB[st][b_k + 4][b_n], Bp + stride4);
        cp_async_ca_16(&sB[st][b_k + 8][b_n], Bp + stride8);
        cp_async_ca_16(&sB[st][b_k + 12][b_n], Bp + stride12);
        cp_async_commit();
    };

    issue_stage(0, A_ptr0, A_ptr1, B_ptr);
    cp_async_wait<0>();
    __syncthreads();

    int cur = 0;
    const int wm_base = warp_m * 32;
    const int wn_base = warp_n * 64;
    const int gid = lane_id >> 2;
    const int tid_g = lane_id & 3;
    const int ldm_a_row = lane_id & 15;
    const int ldm_a_col = (lane_id >> 4) * 4;

    auto compute_tile = [&](int st) {
        #pragma unroll
        for (int kk = 0; kk < 16; kk += 8) {
            uint32_t ra[2][4];
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                ldmatrix_x4(ra[m][0], ra[m][1], ra[m][2], ra[m][3],
                            &sA[st][wm_base + m * 16 + ldm_a_row][kk + ldm_a_col]);
                ra[m][0] = f32_to_tf32(__uint_as_float(ra[m][0]));
                ra[m][1] = f32_to_tf32(__uint_as_float(ra[m][1]));
                ra[m][2] = f32_to_tf32(__uint_as_float(ra[m][2]));
                ra[m][3] = f32_to_tf32(__uint_as_float(ra[m][3]));
            }
            uint32_t rb[2][2];
            rb[0][0] = f32_to_tf32(sB[st][kk + tid_g][wn_base + gid]);
            rb[0][1] = f32_to_tf32(sB[st][kk + tid_g + 4][wn_base + gid]);
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                int rc = n & 1, rn = rc ^ 1;
                if (n < 7) {
                    rb[rn][0] = f32_to_tf32(sB[st][kk + tid_g][wn_base + (n + 1) * 8 + gid]);
                    rb[rn][1] = f32_to_tf32(sB[st][kk + tid_g + 4][wn_base + (n + 1) * 8 + gid]);
                }
                #pragma unroll
                for (int m = 0; m < 2; ++m) {
                    mma_m16n8k8_tf32(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3],
                                     ra[m][0], ra[m][1], ra[m][2], ra[m][3], rb[rc][0], rb[rc][1]);
                }
            }
        }
    };

    for (int bk = k_start + BK; bk < k_end; bk += BK) {
        A_ptr0 += BK; A_ptr1 += BK;
        B_ptr += stride16;
        int nxt = cur ^ 1;
        issue_stage(nxt, A_ptr0, A_ptr1, B_ptr);
        compute_tile(cur);
        cp_async_wait<0>();
        __syncthreads();
        cur = nxt;
    }
    compute_tile(cur);

    float* C_out = (KSPLIT > 1) ? (C + (size_t)ks * N * N) : C;
    #pragma unroll
    for (int m = 0; m < 2; ++m) {
        int r0 = bm + wm_base + m * 16 + gid;
        int r1 = r0 + 8;
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            int c = bn + wn_base + n * 8 + (tid_g << 1);
            *reinterpret_cast<float2*>(&C_out[(size_t)r0 * N + c]) = make_float2(acc[m][n][0], acc[m][n][1]);
            *reinterpret_cast<float2*>(&C_out[(size_t)r1 * N + c]) = make_float2(acc[m][n][2], acc[m][n][3]);
        }
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
        if (size <= 512) {
            float* d_part = g_ws.get(2, 4 * size * size);
            dim3 g(blocks_2d, 4);
            sgemm_tf32_64x128x16<4, 4><<<g, 128>>>(A, B, d_part, N);
            size_t e4 = (size * size) >> 2;
            reduce_splitK<4><<<(unsigned)((e4 + 255) / 256), 256>>>(d_part, C, e4);
        } else if (size == 1024) {
            float* d_part = g_ws.get(2, 2 * size * size);
            dim3 g(blocks_2d, 2);
            sgemm_tf32_64x128x16<4, 2><<<g, 128>>>(A, B, d_part, N);
            size_t e4 = (size * size) >> 2;
            reduce_splitK<2><<<(unsigned)((e4 + 255) / 256), 256>>>(d_part, C, e4);
        } else {
            sgemm_tf32_64x128x16<4, 1><<<blocks_2d, 128>>>(A, B, C, N);
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
