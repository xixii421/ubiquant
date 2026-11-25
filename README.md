# GPU Kernel Optimization Archive

High-performance CUDA C++ kernels developed and tuned for GPU programming competitions (IQuest Competition, etc.).

## Contents

| Contest | Problem | Target GPU | Tuning GPU | Best Result (vs cuBLAS FP32 on L4) | Path |
|---|---|---|---|---|---|
| `practice-2026-11-25` | **A. Matrix Nth Power** ($C = A^n$) | NVIDIA H20 (`sm_90`) | NVIDIA L4 (`sm_89`) | **1.81×** (`3×BF16` TC, `l2_rel = 5.4e-5`) / **1.11×** (Exact FP32) / **2.50×** (`TF32` TC) | [problem-a-matrix-nth-power](iquest/practice-2026-11-25/problem-a-matrix-nth-power/) |
