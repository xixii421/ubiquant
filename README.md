# GPU Kernel Optimization Archive

High-performance CUDA C++ kernels developed and tuned for GPU programming competitions (IQuest Competition, etc.).

## Contents

| Contest | Problem | Judge GPU | Official Score | Official H20 Time | Throughput | Path |
|---|---|---|---|---|---|---|
| `practice-2026-11-25` | **A. Matrix Nth Power** ($C = A^n$) | NVIDIA H20 (`sm_90`) | **1500.0000 / 1500** | **1.63 ms** (vs `4.06652 ms` baseline, **2.49×**) | **68.46 TFLOPS** | [problem-a-matrix-nth-power](iquest/practice-2026-11-25/problem-a-matrix-nth-power/) |
| `practice-2026-11-25` | **C. Minimum Spanning Tree** | NVIDIA H20 (`sm_90`) | **1500.0000 / 1500** | **1.16 ms** (vs `66.7267 ms` baseline, **57.5×**) | — | [problem-c-min-spanning-tree](iquest/practice-2026-11-25/problem-c-min-spanning-tree/) |

## Workspace Skills

- [`iquest-submit`](.agents/skills/iquest-submit/SKILL.md) — Verified browser submission workflow for `competition.iquestlab.com` (Monaco editor atomic insertion, sample Run validation, official H20 Submit & result extraction).
