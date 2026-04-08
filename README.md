# ⚡ Flash Attention Implementation and Performance Study (Backward Pass)

This repository contains CUDA C++ implementations and a performance analysis for the backward pass of the Self-Attention mechanism, focusing on the highly optimized **Flash Attention** technique.

The work was developed as part of a **Topics in Machine Learning (TIML)** course at **National Taiwan University (NTU)**.

---

## 🚀 Project Overview

The primary goal of this project is to demonstrate and profile the drastic performance difference between **naive (textbook)**, **tiled (suboptimal)**, and **optimized Flash Attention** kernels for the Transformer self-attention backward pass.

The analysis is performed using a fixed head dimension D=64 and `half` (FP16) precision for all computations.

### Key Takeaways

1.  **Memory Bottleneck:** Naive implementations are severely limited by memory bandwidth due to reading the large N x N attention matrix up to D times from global memory.
2.  **L2 Thrashing:** For large sequence lengths N >> D, the naive kernel's working set size exceeds the GPU's L2 cache, leading to severe **cache thrashing**.
3.  **Flash Solution:** The optimized kernel eliminates this bottleneck by:
    * **Tiling:** Processing small blocks of Q, K, V and reusing them in fast **Shared Memory**.
    * **Recomputation:** Avoiding the storage and reading of the massive P matrix entirely by recomputing the necessary values on-the-fly.

---

## 📁 Repository Structure

| File Name | Description | Purpose |
| :--- | :--- | :--- |
| `flash_cuda_slow.cu` | **Performance Comparison Kernels** | Benchmarks three kernels against each other to show the performance progression: (1) `naive_backward_kernel` — reads the N×N attention matrix P from global memory D times, extremely slow; (2) `tiled_uncoalesced_kernel` — adds tiling but with broken memory coalescing (strided access); (3) `flash_optimized_half` — full Flash Attention with shared memory caching and P recomputation. Answers: *how much faster is Flash vs naive?* |
| `flash_cuda_flash.cu` | **Block Size Optimization Study** | Takes only the optimized Flash kernel (as a C++ template `<BR, BC>`) and benchmarks it across four tile sizes (4×4, 8×8, 16×16, 32×32) to find the optimal block size for the target GPU. Unlike `flash_cuda_slow.cu`, this file requires no pre-computed P matrix — it recomputes attention weights on-the-fly. Answers: *what is the best tile size for Flash on this GPU?* |
| `check_specs.cu` | **Hardware Analysis** | A utility program that reads and prints key GPU specifications (Compute Capability, L2 Cache Size, Shared Memory per Block) and calculates the **minimum working set size** for the attention problem to justify the memory-bound nature of the naive approach. |
| `README.md` | *This file.* | Project documentation and usage guide. |

---

## ⚙️ Dependencies

* **CUDA Toolkit:** Must be installed (tested with CUDA 11.x+).
* **A modern NVIDIA GPU:** Required to run the CUDA kernels and utilize `__half` types.

## 🛠️ Build and Run Instructions

The files can be compiled using the NVIDIA CUDA compiler (`nvcc`).

### 1. Compile `check_specs.cu`

Run this first to understand your GPU's memory constraints.

```bash
nvcc check_specs.cu -o check_specs
./check_specs
```
This README was generated using Gemini.

---

## Hardware Specifications (Test Machine)

Output of `./check_specs` on the development machine:

```
=== GPU SPECIFICATIONS: NVIDIA GeForce RTX 4090 ===
Compute Capability:       8.9
Global Memory (VRAM):     23.54 GB
L2 Cache Size:            72.00 MB (75497472 bytes)
Shared Mem per Block:     48.00 KB
Registers per Block:      65536

=== ANALYSIS FOR N=16384 ===
Size of ONE tensor (Q):   2.00 MB
Min Working Set (Q+K+V):  6.00 MB
RESULT: Fits in L2 Cache.
```


- Only backward kernels are implemented.