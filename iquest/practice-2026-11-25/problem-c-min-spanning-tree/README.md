# IQuest Competition (`practice-2026-11-25`) — Problem C: Minimum Spanning Tree

Find the total weight of the Minimum Spanning Tree (MST) of a symmetric weighted undirected graph given as an $n \times n$ `float` adjacency matrix $A$ (`A[i][j] > 0` for edges, `0` for absent edges).

- **Judge GPU**: NVIDIA H20 (Hopper `sm_90`, 78 SMs, 4.0 TB/s HBM3, 60 MB L2)
- **Official Baseline (`T_std`)**: `66.7267 ms` across 4 test cases (`n = 1024, 2048, 4096, 6144`)
- **Scoring**: `1000 × min(1.5, T_std / total_GPU_time)` → maximum `1500.0000` points at `≤ 44.484 ms`
- **Best Official Result**: **`1500.0000 / 1500` pts (MAX SCORE, 1 attempt)** · **`1.16 ms`** (`1160.95 μs`) · `4/4 Accepted` ([submission `cmwvn0ogfac3345cpa2im0nhu`](https://competition.iquestlab.com/submissions/cmwvn0ogfac3345cpa2im0nhu))

---

## Files

| File | Description |
|---|---|
| [`solution.cu`](solution.cu) | Winning 1500-pt submission (`v1_boruvka`: parallel Borůvka + strict 64-bit undirected edge key + largest-component row skipping + SMEM lock-free DSU + FP64 weight accumulation) |
| [`benchmark.cu`](benchmark.cu) | Local/Colab verification (17 correctness cases vs CPU FP64 Prim) & multi-distribution benchmark harness |
| [`versions/v1_boruvka.cu`](versions/v1_boruvka.cu) | Archived `v1_boruvka` source |

---

## Official NVIDIA H20 Submission Results

| Test Case | H20 Runtime |
|---|---:|
| `n = 1024` | `144.48 μs` |
| `n = 2048` | `180.02 μs` |
| `n = 4096` | `293.22 μs` |
| `n = 6144` | `543.23 μs` |
| **TOTAL** | **`1.16 ms` (`1160.95 μs`)** |

- **Speedup vs Official Baseline (`66.73 ms`)**: **57.5×**
- **Margin under 1500-pt Threshold (`44.48 ms`)**: **38.3×**

---

## Key Algorithm & Implementation Details

1. **Cycle-Free Parallel Borůvka via Strict 64-bit Edge Key**
   - Every valid undirected edge $(u, v)$ with positive weight $w$ is packed into a single `uint64_t`: `(float_bits(w) << 32) | (min(u,v) << 16) | max(u,v)`.
   - Provides a strict total order identical from both endpoints $(u, v)$ and $(v, u)$, guaranteeing zero cycles even when all edge weights are tied, while enabling native hardware 64-bit `atomicMin`.

2. **Vectorized Coalesced Row Scans & Largest-Component Skipping**
   - Round 0 scans every row with 1 warp per row using 128-bit `float4` coalesced loads (`__ldg`) and warp shuffle reduction (zero atomics).
   - After each merge, the single largest connected component `skip_comp` is excluded from row scanning by the Cut Property — every remaining component still finds its exact minimum outgoing edge and merges, reducing components by at least half per round while skipping the majority of matrix rows in later rounds.
   - Dynamically switches to 1 block (`256` threads) per active row when `num_active <= 256` to keep H20's 78 SMs saturated.

3. **Single-Block Shared-Memory Lock-Free DSU & FP64 Accumulation**
   - Performs lock-free `atomicCAS` union-find (always hooking larger root to smaller root), full path compression, component counting, `skip_comp` selection, active-row compaction, and `double` (FP64) MST weight accumulation (matching PyTorch's Python `float` accumulator bit-for-bit) inside a single CTA in shared memory.

---

## Colab L4 (`sm_89`) Verification & Benchmarks

Correctness suite: **17 / 17 passed** (small sizes, all-equal weights, sparse graphs, path graph, disconnected graphs, two-cliques + heavy bridge, large weights).

| Distribution (`n = 1024, 2048, 4096, 6144`) | L4 Total |
|---|---:|
| Dense uniform `[1, 1000000]` (distinct weights) | `4.269 ms` |
| Dense uniform `[1, 100]` (heavy ties) | `1.876 ms` |
| Sparse 5% density `[1, 10000]` | `4.345 ms` |
| Adversarial two cliques + 1 heavy bridge | `2.656 ms` |

```bash
nvcc -O3 -arch=sm_89 benchmark.cu solution.cu -o bench_mst
./bench_mst
```
