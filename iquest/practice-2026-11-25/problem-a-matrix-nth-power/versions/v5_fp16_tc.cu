#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16_f16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
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

// Ultra-fast vectorized FP32 -> FP16 conversion (8 floats -> 16B STG.128 per thread)
__global__ void fp32_to_fp16_vec8(const float* __restrict__ src, __half* __restrict__ dst, size_t elems8) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elems8) return;
    const float4* s4 = reinterpret_cast<const float4*>(src);
    float4 v0 = s4[idx * 2];
    float4 v1 = s4[idx * 2 + 1];
    __half2 h0 = __float22half2_rn(make_float2(v0.x, v0.y));
    __half2 h1 = __float22half2_rn(make_float2(v0.z, v0.w));
    __half2 h2 = __float22half2_rn(make_float2(v1.x, v1.y));
    __half2 h3 = __float22half2_rn(make_float2(v1.z, v1.w));
    uint4 packed;
    packed.x = *reinterpret_cast<uint32_t*>(&h0);
    packed.y = *reinterpret_cast<uint32_t*>(&h1);
    packed.z = *reinterpret_cast<uint32_t*>(&h2);
    packed.w = *reinterpret_cast<uint32_t*>(&h3);
    reinterpret_cast<uint4*>(dst)[idx] = packed;
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
// High-Throughput FP16 Tensor Core GEMM (64x128x32, 128T, 3-stage cp.async.ca)
// - Pure FP16 inputs -> FP32 accumulation -> FP16 (intermediate) or FP32 (final)
// - Zero-padding XOR-swizzled SMEM (36 KB -> 6 blocks/SM on H20, 0 bank conflicts)
// - ldmatrix.x4 for A + ldmatrix.x4.trans for B (loads 2 N-tiles per instruction!)
// - 4 blocks/SM resident on H20 (78 SMs), 512 blocks on 2048x2048
// ============================================================================
template <int GROUP_M, int KSPLIT, bool OUT_FP16>
__global__ void __launch_bounds__(128, 4)
hgemm_64x128x32_3stg(const __half* __restrict__ A,
                     const __half* __restrict__ B,
                     void* __restrict__ C_void,
                     int N) {
    constexpr int BM = 64, BN = 128, BK = 32, STAGES = 3;
    // 3 * (64*32 + 32*128) * 2 = 36,864 bytes (zero padding!)
    __shared__ __align__(16) __half sA[STAGES][BM][BK];
    __shared__ __align__(16) __half sB[STAGES][BK][BN];

    const int tid = threadIdx.x;
    int tile_m, tile_n;
    swizzle_cta(blockIdx.x, N >> 6, N >> 7, GROUP_M, tile_m, tile_n);
    const int bm = tile_m << 6;
    const int bn = tile_n << 7;

    const int ks = (KSPLIT > 1) ? blockIdx.y : 0;
    const int k_len = (KSPLIT > 1) ? (N / KSPLIT) : N;
    const int k_start = ks * k_len;
    const int num_k = k_len >> 5; // k_len / 32

    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_m = warp_id >> 1; // 0..1 -> 32 rows each
    const int warp_n = warp_id & 1;  // 0..1 -> 64 cols each

    // Global -> SMEM cp.async coordinates (16B = 8 halfs per chunk)
    // A: 64 rows x 4 chunks = 256 chunks / 128 threads = 2 chunks/thread
    const int a_row = tid >> 2;          // 0..31 (+32)
    const int a_chunk = tid & 3;         // 0..3
    const int a_swz0 = (a_chunk ^ ((a_row >> 1) & 3)) << 3;
    const int a_swz1 = (a_chunk ^ (((a_row + 32) >> 1) & 3)) << 3;

    // B: 32 rows x 16 chunks = 512 chunks / 128 threads = 4 chunks/thread
    const int b_row = tid >> 4;          // 0..7 (+8, +16, +24)
    const int b_chunk = tid & 15;        // 0..15
    const int b_swz = (b_chunk ^ (b_row & 7)) << 3; // same for b_row + 8, +16, +24!

    const size_t strideN = (size_t)N;
    const __half* A_base0 = A + (size_t)(bm + a_row) * strideN + k_start + (a_chunk << 3);
    const __half* A_base1 = A_base0 + (strideN << 5);
    const __half* B_base = B + (size_t)(k_start + b_row) * strideN + bn + (b_chunk << 3);

    auto issue_stage = [&](int st, int kt) {
        int bk = kt << 5;
        cp_async_ca_16(&sA[st][a_row][a_swz0], A_base0 + bk);
        cp_async_ca_16(&sA[st][a_row + 32][a_swz1], A_base1 + bk);
        const __half* Bp = B_base + (size_t)bk * strideN;
        cp_async_ca_16(&sB[st][b_row + 0][b_swz], Bp);
        cp_async_ca_16(&sB[st][b_row + 8][b_swz], Bp + (strideN << 3));
        cp_async_ca_16(&sB[st][b_row + 16][b_swz], Bp + (strideN << 4));
        cp_async_ca_16(&sB[st][b_row + 24][b_swz], Bp + strideN * 24);
        cp_async_commit();
    };

    float acc[2][8][4];
    #pragma unroll
    for (int m = 0; m < 2; ++m)
        #pragma unroll
        for (int n = 0; n < 8; ++n)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[m][n][r] = 0.0f;

    // Precompute ldmatrix swizzled offsets for this lane
    const int wm_base = warp_m << 5;
    const int wn_chunk_base = warp_n << 3; // 0 or 8 chunks (= 0 or 64 halfs)

    // For ldmatrix.x4 on A: lane 0..15 -> row = lane & 15, chunk_sub = (lane >> 4) (0 or 1)
    const int ldm_a_row = lane_id & 15;
    const int ldm_a_sub = lane_id >> 4; // 0 or 1
    // Row swizzle bits for m=0 and m=1 (note: adding 16 flips bit 4, does not change (row>>1)&3!)
    const int a_row_m0 = wm_base + ldm_a_row;
    const int a_row_m1 = wm_base + 16 + ldm_a_row;
    const int a_swz_k0 = (ldm_a_sub ^ ((a_row_m0 >> 1) & 3)) << 3;
    const int a_swz_k1 = ((ldm_a_sub + 2) ^ ((a_row_m0 >> 1) & 3)) << 3;

    // For ldmatrix.x4.trans on B:
    // lane 0..7 -> k=0..7, sub=0; lane 8..15 -> k=8..15, sub=0;
    // lane 16..23 -> k=0..7, sub=1; lane 24..31 -> k=8..15, sub=1
    const int ldm_b_k = lane_id & 15;
    const int ldm_b_sub = lane_id >> 4; // 0 or 1
    const int ldm_b_xor = ldm_b_k & 7;

    // Preload stages 0 and 1, then wait for stage 0 to be ready
    issue_stage(0, 0);
    if (num_k > 1) {
        issue_stage(1, 1);
        cp_async_wait<1>();
    } else {
        cp_async_wait<0>();
    }
    __syncthreads();

    int smem_read = 0;
    int smem_write = 2;

    for (int kt = 0; kt < num_k; ++kt) {
        // smem_write was last read at kt-2 and already synchronized -> race-free!
        if (kt + 2 < num_k) {
            issue_stage(smem_write, kt + 2);
        }

        // Compute two K=16 sub-tiles (k_sub = 0 -> cols 0..15, k_sub = 1 -> cols 16..31)
        #pragma unroll
        for (int ks_sub = 0; ks_sub < 2; ++ks_sub) {
            int a_col = (ks_sub == 0) ? a_swz_k0 : a_swz_k1;
            int b_k_row = (ks_sub << 4) + ldm_b_k;

            uint32_t ra[2][4];
            ldmatrix_x4(ra[0][0], ra[0][1], ra[0][2], ra[0][3], &sA[smem_read][a_row_m0][a_col]);
            ldmatrix_x4(ra[1][0], ra[1][1], ra[1][2], ra[1][3], &sA[smem_read][a_row_m1][a_col]);

            uint32_t rb[2][4];
            {
                int b_col = ((wn_chunk_base + ldm_b_sub) ^ ldm_b_xor) << 3;
                ldmatrix_x4_trans(rb[0][0], rb[0][1], rb[0][2], rb[0][3], &sB[smem_read][b_k_row][b_col]);
            }

            #pragma unroll
            for (int np = 0; np < 4; ++np) {
                int rc = np & 1, rn = rc ^ 1;
                if (np < 3) {
                    int b_col = ((wn_chunk_base + (np + 1) * 2 + ldm_b_sub) ^ ldm_b_xor) << 3;
                    ldmatrix_x4_trans(rb[rn][0], rb[rn][1], rb[rn][2], rb[rn][3], &sB[smem_read][b_k_row][b_col]);
                }
                int n0 = np * 2;
                int n1 = n0 + 1;
                #pragma unroll
                for (int m = 0; m < 2; ++m) {
                    mma_m16n8k16_f16(acc[m][n0][0], acc[m][n0][1], acc[m][n0][2], acc[m][n0][3],
                                     ra[m][0], ra[m][1], ra[m][2], ra[m][3], rb[rc][0], rb[rc][1]);
                    mma_m16n8k16_f16(acc[m][n1][0], acc[m][n1][1], acc[m][n1][2], acc[m][n1][3],
                                     ra[m][0], ra[m][1], ra[m][2], ra[m][3], rb[rc][2], rb[rc][3]);
                }
            }
        }

        if (kt + 2 < num_k) {
            cp_async_wait<1>();
            __syncthreads();
        } else if (kt + 1 < num_k) {
            cp_async_wait<0>();
            __syncthreads();
        }

        smem_read = (smem_read + 1 == STAGES) ? 0 : smem_read + 1;
        smem_write = (smem_write + 1 == STAGES) ? 0 : smem_write + 1;
    }

    const int gid = lane_id >> 2;
    const int tid_g = (lane_id & 3) << 1;
    const int wn_base = warp_n << 6;

    if constexpr (KSPLIT == 1 && OUT_FP16) {
        __half* C_h = reinterpret_cast<__half*>(C_void);
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            int r0 = bm + wm_base + m * 16 + gid;
            int r1 = r0 + 8;
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                int c = bn + wn_base + n * 8 + tid_g;
                *reinterpret_cast<__half2*>(&C_h[(size_t)r0 * strideN + c]) =
                    __float22half2_rn(make_float2(acc[m][n][0], acc[m][n][1]));
                *reinterpret_cast<__half2*>(&C_h[(size_t)r1 * strideN + c]) =
                    __float22half2_rn(make_float2(acc[m][n][2], acc[m][n][3]));
            }
        }
    } else {
        float* C_f = reinterpret_cast<float*>(C_void) + ((KSPLIT > 1) ? (size_t)ks * strideN * strideN : 0);
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            int r0 = bm + wm_base + m * 16 + gid;
            int r1 = r0 + 8;
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                int c = bn + wn_base + n * 8 + tid_g;
                *reinterpret_cast<float2*>(&C_f[(size_t)r0 * strideN + c]) = make_float2(acc[m][n][0], acc[m][n][1]);
                *reinterpret_cast<float2*>(&C_f[(size_t)r1 * strideN + c]) = make_float2(acc[m][n][2], acc[m][n][3]);
            }
        }
    }
}

template <int KSPLIT, bool OUT_FP16>
__global__ void reduce_splitK_out(const float* __restrict__ C_part, void* __restrict__ C_out, size_t elems4) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elems4) return;
    const float4* src = reinterpret_cast<const float4*>(C_part);
    float4 sum = src[idx];
    #pragma unroll
    for (int s = 1; s < KSPLIT; ++s) {
        float4 v = src[idx + (size_t)s * elems4];
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
    if constexpr (OUT_FP16) {
        __half2 h0 = __float22half2_rn(make_float2(sum.x, sum.y));
        __half2 h1 = __float22half2_rn(make_float2(sum.z, sum.w));
        uint2 packed;
        packed.x = *reinterpret_cast<uint32_t*>(&h0);
        packed.y = *reinterpret_cast<uint32_t*>(&h1);
        reinterpret_cast<uint2*>(C_out)[idx] = packed;
    } else {
        reinterpret_cast<float4*>(C_out)[idx] = sum;
    }
}

struct Workspace {
    void* buf[4] = {nullptr, nullptr, nullptr, nullptr};
    size_t cap[4] = {0, 0, 0, 0};

    template <typename T>
    T* get(int idx, size_t elems) {
        size_t bytes = elems * sizeof(T);
        if (bytes > cap[idx]) {
            if (buf[idx]) cudaFree(buf[idx]);
            cudaMalloc(&buf[idx], bytes);
            cap[idx] = bytes;
        }
        return reinterpret_cast<T*>(buf[idx]);
    }

    ~Workspace() {
        for (int i = 0; i < 4; ++i) {
            if (buf[i]) cudaFree(buf[i]);
        }
    }
};

static Workspace g_ws;

template <bool OUT_FP16>
static void launch_hgemm(const __half* A, const __half* B, void* C, size_t size) {
    int N = (int)size;
    unsigned blocks_2d = (unsigned)((size >> 6) * (size >> 7));
    if (size == 512) {
        // Split-K=4 -> 128 blocks (saturates all 78 SMs on H20)
        float* d_part = g_ws.get<float>(3, 4 * size * size);
        dim3 g(blocks_2d, 4);
        hgemm_64x128x32_3stg<4, 4, false><<<g, 128>>>(A, B, d_part, N);
        size_t e4 = (size * size) >> 2;
        reduce_splitK_out<4, OUT_FP16><<<(unsigned)((e4 + 255) / 256), 256>>>(d_part, C, e4);
    } else if (size == 1024) {
        // Split-K=2 -> 256 blocks (3.28 blocks/SM on H20)
        float* d_part = g_ws.get<float>(3, 2 * size * size);
        dim3 g(blocks_2d, 2);
        hgemm_64x128x32_3stg<4, 2, false><<<g, 128>>>(A, B, d_part, N);
        size_t e4 = (size * size) >> 2;
        reduce_splitK_out<2, OUT_FP16><<<(unsigned)((e4 + 255) / 256), 256>>>(d_part, C, e4);
    } else {
        // 512 blocks on 2048x2048 (6.56 blocks/SM on H20)
        hgemm_64x128x32_3stg<4, 1, OUT_FP16><<<blocks_2d, 128>>>(A, B, C, N);
    }
}

static void launch_general_gemm(const float* A, const float* B, float* C, size_t size) {
    dim3 block(32, 32);
    dim3 grid((unsigned)((size + 31) / 32), (unsigned)((size + 31) / 32));
    sgemm_general_32<<<grid, block>>>(A, B, C, (int)size);
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

    // Fast FP16 Tensor Core path for size >= 128 && size % 128 == 0
    if ((size & 127) == 0 && size >= 128) {
        __half* h_buf0 = g_ws.get<__half>(0, elems);
        size_t e8 = elems >> 3;
        fp32_to_fp16_vec8<<<(unsigned)((e8 + 255) / 256), 256>>>(input_matrix, h_buf0, e8);

        if ((n & (n - 1)) == 0) {
            int steps = __builtin_ctzll(n);
            if (steps == 1) {
                launch_hgemm<false>(h_buf0, h_buf0, output_matrix, size);
                return;
            }
            __half* h_buf1 = g_ws.get<__half>(1, elems);
            __half* cur = h_buf0;
            __half* nxt = h_buf1;
            for (int s = 0; s < steps - 1; ++s) {
                launch_hgemm<true>(cur, cur, nxt, size);
                std::swap(cur, nxt);
            }
            launch_hgemm<false>(cur, cur, output_matrix, size);
            return;
        }

        // General n >= 3 in FP16 intermediate buffers, final write directly to FP32 output_matrix
        __half* h_buf1 = g_ws.get<__half>(1, elems);
        __half* h_buf2 = g_ws.get<__half>(2, elems);
        bool has_res = false;
        __half* base = h_buf0;
        __half* res = h_buf1;
        __half* scratch = h_buf2;
        size_t p = n;
        while (p > 0) {
            bool need_mul = (p & 1) != 0;
            size_t next_p = p >> 1;
            if (need_mul) {
                if (!has_res) {
                    if (next_p == 0) {
                        // Unreachable for n >= 3, but safe
                        cudaMemcpyAsync(res, base, elems * sizeof(__half), cudaMemcpyDeviceToDevice);
                    } else {
                        cudaMemcpyAsync(res, base, elems * sizeof(__half), cudaMemcpyDeviceToDevice);
                    }
                    has_res = true;
                } else if (next_p == 0) {
                    // Final multiply lands directly into FP32 output_matrix!
                    launch_hgemm<false>(res, base, output_matrix, size);
                    return;
                } else {
                    launch_hgemm<true>(res, base, scratch, size);
                    std::swap(res, scratch);
                }
            }
            p = next_p;
            if (p > 0) {
                // Find buffer != res and != base
                __half* next_base = (h_buf0 != res && h_buf0 != base) ? h_buf0 :
                                    ((h_buf1 != res && h_buf1 != base) ? h_buf1 : h_buf2);
                launch_hgemm<true>(base, base, next_base, size);
                base = next_base;
                scratch = (h_buf0 != res && h_buf0 != base) ? h_buf0 :
                          ((h_buf1 != res && h_buf1 != base) ? h_buf1 : h_buf2);
            }
        }
        return;
    }

    // General fallback for arbitrary matrix sizes (not multiple of 128)
    if ((n & (n - 1)) == 0) {
        int steps = __builtin_ctzll(n);
        if (steps == 1) {
            launch_general_gemm(input_matrix, input_matrix, output_matrix, size);
            return;
        }
        float* tmp0 = g_ws.get<float>(0, elems);
        float* dst = (steps & 1) ? output_matrix : tmp0;
        float* alt = (steps & 1) ? tmp0 : output_matrix;
        launch_general_gemm(input_matrix, input_matrix, dst, size);
        for (int s = 1; s < steps; ++s) {
            launch_general_gemm(dst, dst, alt, size);
            std::swap(dst, alt);
        }
        return;
    }

    float* tmp0 = g_ws.get<float>(0, elems);
    float* tmp1 = g_ws.get<float>(1, elems);
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
                launch_general_gemm(res, base, scratch, size);
                std::swap(res, scratch);
            }
        }
        p >>= 1;
        if (p > 0) {
            float* next_base = (tmp1 != res && tmp1 != base) ? tmp1 :
                               ((tmp0 != res && tmp0 != base) ? tmp0 : output_matrix);
            launch_general_gemm(base, base, next_base, size);
            base = next_base;
            scratch = (output_matrix != res && output_matrix != base) ? output_matrix :
                      ((tmp0 != res && tmp0 != base) ? tmp0 : tmp1);
        }
    }
    if (res != output_matrix) {
        cudaMemcpyAsync(output_matrix, res, bytes, cudaMemcpyDeviceToDevice);
    }
}
