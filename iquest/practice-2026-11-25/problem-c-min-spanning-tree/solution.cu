#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr uint64_t INF_KEY = 0xFFFFFFFFFFFFFFFFULL;

// Pack (weight, min(u,v), max(u,v)) into uint64_t for n <= 65535.
// Positive IEEE-754 float bit pattern preserves strict < ordering.
__device__ __forceinline__ uint64_t pack_edge(float w, uint32_t u, uint32_t v) {
    uint32_t a = (u < v) ? u : v;
    uint32_t b = (u < v) ? v : u;
    return ((uint64_t)__float_as_uint(w) << 32) | ((uint64_t)a << 16) | (uint64_t)b;
}

__device__ __forceinline__ uint64_t warp_reduce_min_u64(uint64_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        uint64_t other = __shfl_down_sync(0xFFFFFFFFu, val, offset);
        if (other < val) val = other;
    }
    return val;
}

// Round 0: every vertex is its own component. One warp per row, vectorized float4 loads.
__global__ void boruvka_round0_kernel(
    const float* __restrict__ A,
    unsigned long long* __restrict__ d_best,
    int n
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;

    for (int row = warp_id; row < n; row += num_warps) {
        const float* row_ptr = A + (size_t)row * (size_t)n;
        uint64_t best = INF_KEY;
        uint32_t u = (uint32_t)row;

        bool aligned = ((((uintptr_t)row_ptr) & 15u) == 0) && ((n & 3) == 0);
        if (aligned) {
            int n4 = n >> 2;
            const float4* row_ptr4 = reinterpret_cast<const float4*>(row_ptr);
            for (int c4 = lane; c4 < n4; c4 += 32) {
                float4 v4 = __ldg(row_ptr4 + c4);
                uint32_t base = (uint32_t)(c4 << 2);
                if (v4.x > 0.0f && base != u) {
                    uint64_t k = pack_edge(v4.x, u, base);
                    if (k < best) best = k;
                }
                if (v4.y > 0.0f && (base + 1u) != u) {
                    uint64_t k = pack_edge(v4.y, u, base + 1u);
                    if (k < best) best = k;
                }
                if (v4.z > 0.0f && (base + 2u) != u) {
                    uint64_t k = pack_edge(v4.z, u, base + 2u);
                    if (k < best) best = k;
                }
                if (v4.w > 0.0f && (base + 3u) != u) {
                    uint64_t k = pack_edge(v4.w, u, base + 3u);
                    if (k < best) best = k;
                }
            }
        } else {
            for (int col = lane; col < n; col += 32) {
                if (col != row) {
                    float w = __ldg(row_ptr + col);
                    if (w > 0.0f) {
                        uint64_t k = pack_edge(w, u, (uint32_t)col);
                        if (k < best) best = k;
                    }
                }
            }
        }

        best = warp_reduce_min_u64(best);
        if (lane == 0) {
            d_best[row] = (unsigned long long)best;
        }
    }
}

// Round >= 1: scan only active rows (vertices not in skip_comp).
// One warp per active row (or multiple warps when num_active is small!).
__global__ void boruvka_active_rows_kernel(
    const float* __restrict__ A,
    const int* __restrict__ d_comp,
    const int* __restrict__ d_active_rows,
    int num_active,
    unsigned long long* __restrict__ d_best,
    int n
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;

    bool n_mul4 = ((n & 3) == 0);

    for (int idx = warp_id; idx < num_active; idx += num_warps) {
        int row = d_active_rows[idx];
        int my_comp = d_comp[row];
        const float* row_ptr = A + (size_t)row * (size_t)n;
        uint64_t best = INF_KEY;
        uint32_t u = (uint32_t)row;

        if (n_mul4 && ((((uintptr_t)row_ptr) & 15u) == 0)) {
            int n4 = n >> 2;
            const float4* row_ptr4 = reinterpret_cast<const float4*>(row_ptr);
            const int4* comp4 = reinterpret_cast<const int4*>(d_comp);
            for (int c4 = lane; c4 < n4; c4 += 32) {
                int4 cc = __ldg(comp4 + c4);
                if (cc.x == my_comp && cc.y == my_comp && cc.z == my_comp && cc.w == my_comp) {
                    continue;
                }
                float4 v4 = __ldg(row_ptr4 + c4);
                uint32_t base = (uint32_t)(c4 << 2);
                if (cc.x != my_comp && v4.x > 0.0f) {
                    uint64_t k = pack_edge(v4.x, u, base);
                    if (k < best) best = k;
                }
                if (cc.y != my_comp && v4.y > 0.0f) {
                    uint64_t k = pack_edge(v4.y, u, base + 1u);
                    if (k < best) best = k;
                }
                if (cc.z != my_comp && v4.z > 0.0f) {
                    uint64_t k = pack_edge(v4.z, u, base + 2u);
                    if (k < best) best = k;
                }
                if (cc.w != my_comp && v4.w > 0.0f) {
                    uint64_t k = pack_edge(v4.w, u, base + 3u);
                    if (k < best) best = k;
                }
            }
        } else {
            for (int col = lane; col < n; col += 32) {
                if (__ldg(d_comp + col) != my_comp) {
                    float w = __ldg(row_ptr + col);
                    if (w > 0.0f) {
                        uint64_t k = pack_edge(w, u, (uint32_t)col);
                        if (k < best) best = k;
                    }
                }
            }
        }

        best = warp_reduce_min_u64(best);
        if (lane == 0 && best != INF_KEY) {
            atomicMin(&d_best[my_comp], (unsigned long long)best);
        }
    }
}

// When num_active is small (e.g. <= 256), split each row across multiple warps (1 block per row)
// so all SMs stay busy!
__global__ void boruvka_active_rows_block_kernel(
    const float* __restrict__ A,
    const int* __restrict__ d_comp,
    const int* __restrict__ d_active_rows,
    int num_active,
    unsigned long long* __restrict__ d_best,
    int n
) {
    int idx = blockIdx.x;
    if (idx >= num_active) return;

    int row = d_active_rows[idx];
    int my_comp = d_comp[row];
    const float* row_ptr = A + (size_t)row * (size_t)n;
    uint64_t best = INF_KEY;
    uint32_t u = (uint32_t)row;

    bool aligned = ((((uintptr_t)row_ptr) & 15u) == 0) && ((n & 3) == 0);
    if (aligned) {
        int n4 = n >> 2;
        const float4* row_ptr4 = reinterpret_cast<const float4*>(row_ptr);
        const int4* comp4 = reinterpret_cast<const int4*>(d_comp);
        for (int c4 = threadIdx.x; c4 < n4; c4 += blockDim.x) {
            int4 cc = __ldg(comp4 + c4);
            if (cc.x == my_comp && cc.y == my_comp && cc.z == my_comp && cc.w == my_comp) {
                continue;
            }
            float4 v4 = __ldg(row_ptr4 + c4);
            uint32_t base = (uint32_t)(c4 << 2);
            if (cc.x != my_comp && v4.x > 0.0f) {
                uint64_t k = pack_edge(v4.x, u, base);
                if (k < best) best = k;
            }
            if (cc.y != my_comp && v4.y > 0.0f) {
                uint64_t k = pack_edge(v4.y, u, base + 1u);
                if (k < best) best = k;
            }
            if (cc.z != my_comp && v4.z > 0.0f) {
                uint64_t k = pack_edge(v4.z, u, base + 2u);
                if (k < best) best = k;
            }
            if (cc.w != my_comp && v4.w > 0.0f) {
                uint64_t k = pack_edge(v4.w, u, base + 3u);
                if (k < best) best = k;
            }
        }
    } else {
        for (int col = threadIdx.x; col < n; col += blockDim.x) {
            if (__ldg(d_comp + col) != my_comp) {
                float w = __ldg(row_ptr + col);
                if (w > 0.0f) {
                    uint64_t k = pack_edge(w, u, (uint32_t)col);
                    if (k < best) best = k;
                }
            }
        }
    }

    best = warp_reduce_min_u64(best);
    __shared__ uint64_t s_warp[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) s_warp[wid] = best;
    __syncthreads();

    if (wid == 0) {
        int nwarps = blockDim.x >> 5;
        uint64_t v = (lane < nwarps) ? s_warp[lane] : INF_KEY;
        v = warp_reduce_min_u64(v);
        if (lane == 0 && v != INF_KEY) {
            atomicMin(&d_best[my_comp], (unsigned long long)v);
        }
    }
}

__device__ __forceinline__ int find_root_smem(volatile int* parent, int x) {
    while (true) {
        int p = parent[x];
        if (p == x) return x;
        x = p;
    }
}

__device__ __forceinline__ bool union_smem(int* parent, int u, int v) {
    volatile int* vp = parent;
    while (true) {
        u = find_root_smem(vp, u);
        v = find_root_smem(vp, v);
        if (u == v) return false;
        if (u < v) { int t = u; u = v; v = t; } // u > v
        int old = atomicCAS(&parent[u], u, v);
        if (old == u) return true;
    }
}

struct MetaState {
    int num_comps;
    int num_active;
    int skip_comp;
    int disconnected;
    double total_weight;
};

// Single-block merge + path compression + skip_comp selection + active_rows compaction.
// Uses dynamic shared memory: 2 * n * sizeof(int) bytes (for n=6144 -> 49,152 bytes).
// Supports up to n = 12288 on Hopper/Ada with 98KB dynamic SMEM (or static 48KB for n <= 6144).
__global__ void boruvka_merge_smem_kernel(
    int* __restrict__ d_comp,
    unsigned long long* __restrict__ d_best,
    int* __restrict__ d_active_rows,
    MetaState* __restrict__ d_meta,
    float* __restrict__ min_weight,
    int n,
    int is_round0
) {
    extern __shared__ int smem[];
    int* s_parent = smem;          // [n]
    int* s_size   = smem + n;      // [n]

    __shared__ int s_disconnected;
    __shared__ int s_active_cnt;
    __shared__ int s_num_comps;
    __shared__ int s_skip_comp;
    __shared__ double s_warp_w[32];
    __shared__ int s_warp_max_sz[32];
    __shared__ int s_warp_max_id[32];
    __shared__ int s_warp_cnt[32];

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    int nwarps = bdim >> 5;

    int prev_skip = is_round0 ? -1 : d_meta->skip_comp;

    // 1. Load current component roots into s_parent, clear s_size
    if (tid == 0) {
        s_disconnected = 0;
        s_active_cnt = 0;
    }
    for (int i = tid; i < n; i += bdim) {
        s_parent[i] = is_round0 ? i : d_comp[i];
        s_size[i] = 0;
    }
    __syncthreads();

    // Mark which indices were active roots before merging (to detect disconnected components)
    // A root r is active iff s_parent[r] == r && r != prev_skip.
    // Wait: we must snapshot is_root BEFORE unions modify s_parent!
    // We can use s_size[i] temporarily as a flag (1 = was active root before union).
    for (int i = tid; i < n; i += bdim) {
        if (s_parent[i] == i && i != prev_skip) {
            s_size[i] = 1;
        }
    }
    __syncthreads();

    // 2. Perform parallel lock-free DSU unions on every active root's best outgoing edge
    double local_w = 0.0;
    for (int i = tid; i < n; i += bdim) {
        if (s_size[i] == 1) {
            uint64_t key = (uint64_t)d_best[i];
            if (key == INF_KEY) {
                s_disconnected = 1;
            } else {
                int a = (int)((key >> 16) & 0xFFFFu);
                int b = (int)(key & 0xFFFFu);
                if (union_smem(s_parent, a, b)) {
                    float w = __uint_as_float((uint32_t)(key >> 32));
                    local_w += (double)w;
                }
            }
        }
    }
    __syncthreads();

    // 3. Reduce local_w in FP64
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_w += __shfl_down_sync(0xFFFFFFFFu, local_w, offset);
    }
    if (lane == 0) s_warp_w[wid] = local_w;
    __syncthreads();

    if (wid == 0) {
        double w = (lane < nwarps) ? s_warp_w[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            w += __shfl_down_sync(0xFFFFFFFFu, w, offset);
        }
        if (lane == 0) {
            double total = (is_round0 ? 0.0 : d_meta->total_weight) + w;
            d_meta->total_weight = total;
            d_meta->disconnected = s_disconnected;
        }
    }

    // Clear s_size for component size counting
    for (int i = tid; i < n; i += bdim) {
        s_size[i] = 0;
    }
    __syncthreads();

    if (s_disconnected) {
        if (tid == 0) {
            d_meta->num_comps = 0;
            d_meta->num_active = 0;
            *min_weight = -1.0f;
        }
        return;
    }

    // 4. Flatten DSU -> write d_comp[i] and count component sizes
    volatile int* vp = s_parent;
    for (int i = tid; i < n; i += bdim) {
        int r = find_root_smem(vp, i);
        d_comp[i] = r;
        atomicAdd(&s_size[r], 1);
    }
    __syncthreads();

    // 5. Count number of components and find largest component (skip_comp)
    int local_cnt = 0;
    int best_sz = -1;
    int best_id = -1;
    for (int i = tid; i < n; i += bdim) {
        int sz = s_size[i];
        if (sz > 0) {
            local_cnt++;
            if (sz > best_sz) {
                best_sz = sz;
                best_id = i;
            }
        }
    }

    // Warp reduction for count and argmax
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_cnt += __shfl_down_sync(0xFFFFFFFFu, local_cnt, offset);
        int osz = __shfl_down_sync(0xFFFFFFFFu, best_sz, offset);
        int oid = __shfl_down_sync(0xFFFFFFFFu, best_id, offset);
        if (osz > best_sz || (osz == best_sz && oid < best_id)) {
            best_sz = osz;
            best_id = oid;
        }
    }
    if (lane == 0) {
        s_warp_cnt[wid] = local_cnt;
        s_warp_max_sz[wid] = best_sz;
        s_warp_max_id[wid] = best_id;
    }
    __syncthreads();

    if (wid == 0) {
        int c = (lane < nwarps) ? s_warp_cnt[lane] : 0;
        int bsz = (lane < nwarps) ? s_warp_max_sz[lane] : -1;
        int bid = (lane < nwarps) ? s_warp_max_id[lane] : -1;
        for (int offset = 16; offset > 0; offset >>= 1) {
            c += __shfl_down_sync(0xFFFFFFFFu, c, offset);
            int osz = __shfl_down_sync(0xFFFFFFFFu, bsz, offset);
            int oid = __shfl_down_sync(0xFFFFFFFFu, bid, offset);
            if (osz > bsz || (osz == bsz && oid < bid)) {
                bsz = osz;
                bid = oid;
            }
        }
        if (lane == 0) {
            s_num_comps = c;
            s_skip_comp = bid;
            d_meta->num_comps = c;
            d_meta->skip_comp = bid;
            if (c <= 1) {
                d_meta->num_active = 0;
                *min_weight = (float)(d_meta->total_weight);
            }
        }
    }
    __syncthreads();

    if (s_num_comps <= 1) return;

    // 6. Reset d_best for active roots and compact d_active_rows
    int skip = s_skip_comp;
    for (int i = tid; i < n; i += bdim) {
        if (s_size[i] > 0 && i != skip) {
            d_best[i] = INF_KEY;
        }
        int c = d_comp[i];
        if (c != skip) {
            int pos = atomicAdd(&s_active_cnt, 1);
            d_active_rows[pos] = i;
        }
    }
    __syncthreads();

    if (tid == 0) {
        d_meta->num_active = s_active_cnt;
    }
}

__global__ void write_scalar_kernel(float* out, float val) {
    *out = val;
}

struct Workspace {
    int cap = 0;
    int* d_comp = nullptr;
    unsigned long long* d_best = nullptr;
    int* d_active_rows = nullptr;
    MetaState* d_meta = nullptr;
    MetaState* h_meta = nullptr;

    void ensure(int n) {
        if (n <= cap && d_meta != nullptr) return;
        if (d_comp) cudaFree(d_comp);
        if (d_best) cudaFree(d_best);
        if (d_active_rows) cudaFree(d_active_rows);
        if (d_meta) cudaFree(d_meta);
        if (h_meta) cudaFreeHost(h_meta);

        int new_cap = (n < 6144) ? 6144 : n;
        cudaMalloc(&d_comp, (size_t)new_cap * sizeof(int));
        cudaMalloc(&d_best, (size_t)new_cap * sizeof(unsigned long long));
        cudaMalloc(&d_active_rows, (size_t)new_cap * sizeof(int));
        cudaMalloc(&d_meta, sizeof(MetaState));
        cudaMallocHost(&h_meta, sizeof(MetaState));
        cap = new_cap;
    }
};

static Workspace g_ws;

} // namespace

// Note: A, min_weight are device pointers
extern "C" void solution(const float* A, float* min_weight, size_t n_size) {
    int n = (int)n_size;
    if (n <= 1) {
        write_scalar_kernel<<<1, 1>>>(min_weight, 0.0f);
        return;
    }

    g_ws.ensure(n);

    size_t smem_bytes = (size_t)2 * (size_t)n * sizeof(int);
    static size_t max_configured_smem = 32 * 1024;
    if (smem_bytes > max_configured_smem) {
        cudaFuncSetAttribute(
            boruvka_merge_smem_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes
        );
        max_configured_smem = smem_bytes;
    }

    // Round 0: full coalesced scan, 1 warp per row
    {
        int threads = 256; // 8 warps per block
        int warps_per_block = threads >> 5;
        int blocks = (n + warps_per_block - 1) / warps_per_block;
        boruvka_round0_kernel<<<blocks, threads>>>(A, g_ws.d_best, n);
        boruvka_merge_smem_kernel<<<1, 1024, smem_bytes>>>(
            g_ws.d_comp, g_ws.d_best, g_ws.d_active_rows, g_ws.d_meta, min_weight, n, 1
        );
    }

    // Subsequent rounds
    for (int round = 1; round < 32; ++round) {
        cudaMemcpy(g_ws.h_meta, g_ws.d_meta, sizeof(MetaState), cudaMemcpyDeviceToHost);
        if (g_ws.h_meta->disconnected || g_ws.h_meta->num_comps <= 1) {
            break;
        }
        int num_active = g_ws.h_meta->num_active;
        if (num_active <= 256) {
            boruvka_active_rows_block_kernel<<<num_active, 256>>>(
                A, g_ws.d_comp, g_ws.d_active_rows, num_active, g_ws.d_best, n
            );
        } else {
            int threads = 256;
            int warps_per_block = threads >> 5;
            int blocks = (num_active + warps_per_block - 1) / warps_per_block;
            boruvka_active_rows_kernel<<<blocks, threads>>>(
                A, g_ws.d_comp, g_ws.d_active_rows, num_active, g_ws.d_best, n
            );
        }
        boruvka_merge_smem_kernel<<<1, 1024, smem_bytes>>>(
            g_ws.d_comp, g_ws.d_best, g_ws.d_active_rows, g_ws.d_meta, min_weight, n, 0
        );
    }
}
