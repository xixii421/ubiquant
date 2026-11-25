# IQuest Competition (`practice-2026-11-25`) — Problem A: Matrix Nth Power

Compute $C = A^n$ for a square FP32 row-major matrix $A \in \mathbb{R}^{\text{size} \times \text{size}}$ and non-negative integer $n$.

- **Judge GPU**: NVIDIA H20 (Hopper `sm_90`, 78 SMs, 4.0 TB/s HBM3, 60 MB L2, 228 KB SMEM/SM, 44 TFLOPS FP32 / 74 TFLOPS TF32 / 148 TFLOPS FP16/BF16)
- **Official Baseline (`T_std`)**: `4.06652 ms` across 9 test cases (`512, 1024, 2048` × `power = 2, 4, 8`; total `117.57 GFLOPs`)
- **Scoring**: `1000 × min(1.5, T_std / total_GPU_time)` → maximum `1500.0000` points at `≤ 2.711 ms`
- **Best Official Result**: **`1500.0000 / 1500` pts (MAX SCORE)** · **`1.63 ms`** · **`68,458.97 GFLOPS`** · `9/9 Accepted` ([submission `cmwvm9lw6a5ctymcuxinva2jj`](https://competition.iquestlab.com/submissions/cmwvm9lw6a5ctymcuxinva2jj))

---

## Files

| File | Description |
|---|---|
| [`solution.cu`](solution.cu) | **Winning 1500-pt submission** (`v5_fp16_tc`: 3-stage `cp.async.ca` + XOR-swizzled FP16 Tensor Core, `1.63 ms` on H20) |
| [`benchmark.cu`](benchmark.cu) | Local/Colab verification (29 edge cases + 9 official cases) & benchmark harness vs cuBLAS FP32 / TF32 |
| [`versions/v0_naive.cu`](versions/v0_naive.cu) | Baseline: 32×32 shared-memory tiled SGEMM + zero-copy binary exponentiation |
| [`versions/v1_regtile.cu`](versions/v1_regtile.cu) | 128×128×8 double-buffered register-tiled FP32 SGEMM + CTA swizzle |
| [`versions/v2_fp32_simt.cu`](versions/v2_fp32_simt.cu) | **Exact IEEE-754 FP32** (`64×128×16`, `cp.async.ca`, Split-K=4) — 1.11× faster than cuBLAS FP32 on L4 (`l2_rel = 2.29e-6`) |
| [`versions/v3_bf16_3x_tc.cu`](versions/v3_bf16_3x_tc.cu) | **3×BF16 Tensor Core** (`64×128×16`, `mma.sync.m16n8k16.bf16`) — `4.53 ms` / `897.31 pts` on H20 (`l2_rel = 5.40e-5`) |
| [`versions/v4_tf32_tc.cu`](versions/v4_tf32_tc.cu) | **Pure-PTX TF32 Tensor Core** (`64×128×16`, `cp.async.ca`, `mma.sync.m16n8k8.tf32`) — `3.22 ms` / `1261.53 pts` on H20 |
| [`versions/v5_fp16_tc.cu`](versions/v5_fp16_tc.cu) | **FP16 Tensor Core (3-stage `cp.async.ca` + XOR swizzle)** — **`1.63 ms` / `1500.0000 pts` on H20** (promoted to `solution.cu`) |

---

## Official NVIDIA H20 Submission Results

All three Tensor Core submissions passed **9/9** test cases on NVIDIA H20 (`nvcc CUDA 13.0.2`):

| Submission | Kernel | H20 Total Time | Throughput | Score | Status |
|---|---|---:|---:|---:|---|
| `cmwvk684z9pyss1dv8zorpfgb` | `v3_bf16_3x_tc.cu` | 4.53 ms | 30.20 TFLOPS | 897.3112 | Accepted (9/9) |
| `cmwvk9d9p9r0pyxqhpcbhghsj` | `v4_tf32_tc.cu` | 3.22 ms | 41.21 TFLOPS | 1261.5299 | Accepted (9/9) |
| **`cmwvm9lw6a5ctymcuxinva2jj`** | **`solution.cu` (`v5_fp16_tc.cu`)** | **1.63 ms** | **68.46 TFLOPS** | **1500.0000** | **Accepted (9/9)** |

### Per-Case H20 Runtime Breakdown

| Test Case | `v3_bf16_3x_tc` | `v4_tf32_tc` | **`solution.cu` (`v5_fp16_tc`)** | Speedup (`v5` vs `v4`) |
|---|---:|---:|---:|---:|
| `512×512 power=2` | 19.74 μs | 15.54 μs | **13.54 μs** | 1.15× |
| `512×512 power=4` | 38.96 μs | 30.59 μs | **23.78 μs** | 1.29× |
| `512×512 power=8` | 58.05 μs | 45.22 μs | **33.92 μs** | 1.33× |
| `1024×1024 power=2` | 101.76 μs | 72.05 μs | **42.21 μs** | 1.71× |
| `1024×1024 power=4` | 202.85 μs | 143.63 μs | **79.12 μs** | 1.82× |
| `1024×1024 power=8` | 304.98 μs | 215.20 μs | **116.32 μs** | 1.85× |
| `2048×2048 power=2` | 634.63 μs | 450.83 μs | **223.91 μs** | 2.01× |
| `2048×2048 power=4` | 1.27 ms | 899.88 μs | **438.69 μs** | 2.05× |
| `2048×2048 power=8` | 1.90 ms | 1.35 ms | **653.89 μs** | 2.06× |
| **TOTAL** | **4.53 ms** | **3.22 ms** | **1.63 ms** | **1.98×** |

---

## Key Architecture & Optimizations in `solution.cu` (`v5_fp16_tc.cu`)

1. **FP16 Tensor Core (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`) with FP32 Accumulation**
   - Unlocks H20's **148 TFLOPS FP16 Tensor Core** (2.0× TF32's 74 TFLOPS) while preserving the same 10-bit explicit mantissa precision as TF32 and accumulating in FP32.
   - Input matrix $A$ is converted once via a vectorized `fp32_to_fp16_vec8` kernel (`2×float4 → uint4` coalesced `STG.128`).
   - Intermediate powers ($A^2, A^4$) are written directly in FP16 (`OUT_FP16 = true`) to persistent device workspace buffers (halving memory traffic and eliminating all `cvt` inside the GEMM K-loop); only the final squaring writes FP32 directly into `output_matrix`.

2. **3-Stage `cp.async.ca` Pipeline + Zero-Padding XOR-Swizzled SMEM**
   - Tile `64×128×32` (128 threads = `2×2` warps, 4 blocks/SM on H20) uses `36 KB` shared memory (`STAGES = 3`) with **zero padding** and **zero bank conflicts** on both `cp.async.ca` writes and `ldmatrix` reads:
     - `sA[3][64][32]`: `swz_chunk = chunk ^ ((row >> 1) & 3)`
     - `sB[3][32][128]`: `swz_chunk = chunk ^ (row & 7)`
   - Uses `ldmatrix.sync.aligned.m8n8.x4.shared.b16` for `A` and `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` for `B` (loading two `16×8` B-tiles per instruction with double-buffered `rb` fragments), with a single `__syncthreads()` per `BK=32` K-tile.

3. **H20 78-SM Wave Balancing & General Edge-Case Coverage**
   - `2048×2048`: `512` CTAs (6.56 blocks/SM on H20) with L2-friendly CTA swizzling (`GROUP_M = 4`).
   - `512×512` (`Split-K = 4` → 128 CTAs) and `1024×1024` (`Split-K = 2` → 256 CTAs) with fused FP16/FP32 vectorized `float4` reduction epilogue saturate all 78 SMs on H20.
   - Handles `n = 0` ($I$), `n = 1` ($A$), arbitrary general $n$ via binary exponentiation, and arbitrary non-multiple-of-128 matrix sizes via a bounds-checked fallback kernel.

---

## Colab L4 (`sm_89`) Development Benchmarks (Reference Progression)

| Version | L4 Total (9 cases) | Avg TFLOPS | vs cuBLAS FP32 | `2048 p=8` `l2_rel` |
|---|---:|---:|---:|---:|
| `v0_naive` | 74.6465 ms | 1.58 | 0.11× | `2.29e-6` |
| `v1_regtile` | 8.8995 ms | 13.21 | 1.01× | `2.29e-6` |
| cuBLAS FP32 (`PEDANTIC`) | 8.37–9.56 ms | 12.3–14.1 | 1.00× | `0` |
| `v2_fp32_simt` | 8.5719 ms | 13.72 | 1.11× | `2.29e-6` |
| `v3_bf16_3x_tc` | 4.9725 ms | 23.64 | 1.81× | `5.40e-5` |
| cuBLAS TF32 | 4.04–4.15 ms | 28.3–29.1 | 2.24× | `7.96e-4` |
| `v4_tf32_tc` | 3.6157 ms | 32.52 | 2.50× | `7.96e-4` |

---

## Build & Run Locally / on Colab

```bash
nvcc -O3 -arch=sm_89 benchmark.cu solution.cu -lcublas -o bench_solution
./bench_solution
```
