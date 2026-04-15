#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Configuration
#define D 64         // Head Dimension
#define BR 16        // Block Row Size 
#define BC 16        // Block Col Size

#define CHECK_CUDA(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Helper: Atomic Add for Half
#if __CUDA_ARCH__ < 700
__device__ void atomicAddHalf(half* address, half val) {
    unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        half h_old = (size_t)address & 2 ? __ushort_as_half(old >> 16) : __ushort_as_half(old & 0xffff);
        half h_sum = __hadd(h_old, val);
        unsigned int next = (size_t)address & 2 
            ? (old & 0xffff) | (__half_as_ushort(h_sum) << 16) 
            : (old & 0xffff0000) | __half_as_ushort(h_sum);
        old = atomicCAS(address_as_ui, assumed, next);
    } while (assumed != old);
}
#else
__device__ void atomicAddHalf(half* address, half val) {
    atomicAdd(address, val);
}
#endif

// Helper: Cache Flush Kernel
__global__ void flush_l2_cache_kernel(float* buffer, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buffer[idx] += 1.0f;
    }
}

// Regular Attntion
// Reads P from memory 64 times per pixel. Extremely slow.
__global__ void naive_backward_kernel(
    const half* Q, const half* K, const half* V, 
    const half* P, 
    const half* dO, const float* Delta,
    half* dQ, half* dK, half* dV,
    int n, int d, float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (idx < n) {
        // dV
        for (int x = 0; x < d; x++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += __half2float(P[i * n + idx]) * __half2float(dO[i * d + x]);
            }
            dV[idx * d + x] = __float2half(sum);
        }
        // both dP and dS are computed twice for dQ and dK
        // dQ
        for (int x = 0; x < d; x++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                float dP = 0.0f;
                for (int xx = 0; xx < d; xx++) dP += __half2float(dO[idx * d + xx]) * __half2float(V[j * d + xx]);
                float val_S = __half2float(P[idx * n + j]) * (dP - Delta[idx]) * scale;
                sum += val_S * __half2float(K[j * d + x]);
            }
            dQ[idx * d + x] = __float2half(sum);
        }
        // dK
        for (int x = 0; x < d; x++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                float dP = 0.0f;
                for (int xx = 0; xx < d; xx++) dP += __half2float(dO[i * d + xx]) * __half2float(V[idx * d + xx]);
                float val_S = __half2float(P[i * n + idx]) * (dP - Delta[i]) * scale;
                sum += val_S * __half2float(Q[i * d + x]);
            }
            dK[idx * d + x] = __float2half(sum);
        }
    }
}

// Uncoalesced Tiled Attention
// Tiled logic, but breaks memory coalescing.
__global__ void tiled_uncoalesced_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, 
    const half* __restrict__ P, 
    const half* __restrict__ dO, 
    half* __restrict__ dQ, half* __restrict__ dK, half* __restrict__ dV, 
    const float* __restrict__ Delta, float scale, int n
) {
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int row_q_start = blockIdx.x * BR; 
    
    // This causes strided memory access.
    int q_idx = row_q_start + tx; 

    for (int j_start = 0; j_start < n; j_start += BC) {
        for (int k_row = 0; k_row < BC; k_row++) {
            int global_k = j_start + k_row;
            
            float val_f = __half2float(P[q_idx * n + global_k]); 
            
            float dP_val = 0.0f;
            for (int x = 0; x < D; x++) {
                dP_val += __half2float(dO[q_idx * D + x]) * __half2float(V[global_k * D + x]);
            }

            float dS_f = val_f * (dP_val - Delta[q_idx]) * scale;

            for (int x = ty; x < D; x += BC) {
                float val_K = __half2float(K[global_k * D + x]);
                atomicAddHalf(&dQ[q_idx * D + x], __float2half(dS_f * val_K));

                float val_Q = __half2float(Q[q_idx * D + x]);
                atomicAddHalf(&dK[global_k * D + x], __float2half(dS_f * val_Q));

                float val_dO = __half2float(dO[q_idx * D + x]);
                atomicAddHalf(&dV[global_k * D + x], __float2half(val_f * val_dO));
            }
        }
    }
}

// Flash Attention
// Fully coalesced, Shared Memory caching, and Recomputation.
__global__ void flash_optimized_half(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, 
    const half* __restrict__ dO, const float* __restrict__ L, const float* __restrict__ m, 
    half* __restrict__ dQ, half* __restrict__ dK, half* __restrict__ dV, 
    const float* __restrict__ Delta, float scale, int n
) {
    __shared__ half sQ[BR * D];
    __shared__ half sdO[BR * D];
    __shared__ half sK[BC * D];
    __shared__ half sV[BC * D];

    int tx = threadIdx.x; int ty = threadIdx.y; 
    int row_q_start = blockIdx.x * BR; int q_idx = row_q_start + ty;

    if (row_q_start < n) {
        for (int x = tx; x < D; x += BC) {
            sQ[ty * D + x] = Q[row_q_start * D + x];
            sdO[ty * D + x] = dO[row_q_start * D + x];
        }
    }
    __syncthreads();

    for (int j_start = 0; j_start < n; j_start += BC) {
        for (int x = tx; x < D; x += BC) {
            sK[ty * D + x] = K[(j_start + ty) * D + x];
            sV[ty * D + x] = V[(j_start + ty) * D + x];
        }
        __syncthreads();

        for (int k_row = 0; k_row < BC; k_row++) {
            int global_k = j_start + k_row;
            float dot = 0.0f;
            for (int x = 0; x < D; x++) dot += __half2float(sQ[ty * D + x]) * __half2float(sK[k_row * D + x]);
            
            // Recomputation of P:
            float val_f = expf(dot * scale - m[q_idx]) / L[q_idx];
            float dP_val = 0.0f;
            for (int x = 0; x < D; x++) dP_val += __half2float(sdO[ty * D + x]) * __half2float(sV[k_row * D + x]);

            float dS_f = val_f * (dP_val - Delta[q_idx]) * scale;

            for (int x = tx; x < D; x += BC) {
                atomicAddHalf(&dQ[q_idx * D + x], __float2half(dS_f * __half2float(sK[k_row * D + x])));
                atomicAddHalf(&dK[global_k * D + x], __float2half(dS_f * __half2float(sQ[ty * D + x])));
                atomicAddHalf(&dV[global_k * D + x], __float2half(val_f * __half2float(sdO[ty * D + x])));
            }
        }
        __syncthreads();
    }
}

// Helpers
__global__ void compute_full_P_half(const half* Q, const half* K, half* P, const float* L, const float* m, int n, int d, float scale) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    if (row < n && col < n) {
        float dot = 0.0f;
        for (int x = 0; x < d; x++) dot += __half2float(Q[row * d + x]) * __half2float(K[col * d + x]);
        P[row * n + col] = __float2half(expf(dot * scale - m[row]) / L[row]);
    }
}
// Helper: Convert float to half
__global__ void float2half_kernel(half* out, const float* in, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = __float2half(in[i]);
}
__global__ void compute_delta_half(const half* dO, const half* O, float* Delta, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int x = 0; x < d; x++) sum += __half2float(dO[i * d + x]) * __half2float(O[i * d + x]);
        Delta[i] = sum;
    }
}

// ---------------------------------------------------------------
// Naive Forward (Shared Memory = M, ref.tex block-wise)
//
// Q_{I_i}, K_{J_j}, V_{J_j} tiles all live in SRAM (main memory M).
// Tile sizes (with BC = BR = 32, D = 64):
//   sQ : BR × D = 32 × 64 × 2 B = 4 KB
//   sK : BC × D = 32 × 64 × 2 B = 4 KB
//   sV : BC × D = 32 × 64 × 2 B = 4 KB   total: 12 KB < 48 KB SRAM
//
// Q_{I_i} is loaded from HBM ONCE per I-block (outer loop), then stays in
// SRAM while all J-blocks are processed — matching the ref.tex model where
// Q_{I_i} is placed in main memory M and never reloaded.
//
// Three passes (scale and max omitted — does not affect memory traffic):
//   Pass 1: compute P_{I_i,J_j} = exp(Q K^T), write to P (HBM secondary memory)
//   Pass 2: read P from HBM, divide by row sum, write normalised P back
//   Pass 3: stream V tiles through SRAM, read P from HBM, accumulate O
//
// HBM traffic per I-block (secondary memory accesses):
//   Q  : 1 × BRd   — loaded once into sQ at start of I-block
//   K  : T_c × BCd — one HBM load per J-block into sK
//   P  : 3 × BRN   — write exp scores, read+write normalise, read for O
//   V  : T_c × BCd — one HBM load per J-block into sV
//   O  : 1 × BRd   — written once at end
//
// Notation follows ref.tex:
//   P_{I_i, J_j}  — attention score tile (written to HBM)
//   O_{I_i,:}     — output tile (accumulated in registers, written once)
// ---------------------------------------------------------------
__global__ void naive_smem_forward_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
    half* __restrict__ P_mat, half* __restrict__ O,
    float* __restrict__ L,
    int n
) {
    __shared__ half sQ[BR * D];
    __shared__ half sK[BC * D];
    __shared__ half sV[BC * D];

    int tx = threadIdx.x, ty = threadIdx.y;
    int q_idx = blockIdx.x * BR + ty;
    if (q_idx >= n) return;

    int tid = ty * BC + tx;  // linear index within the (BC × BR) thread block

    // ---- Load Q_{I_i} into sQ once — stays in SRAM for all J-blocks ----
    int q_block_start = blockIdx.x * BR;
    for (int elem = tid; elem < BR * D; elem += BR * BC)
        sQ[elem] = Q[q_block_start * D + elem];   // sQ[li*D+x] = Q[(q_block_start+li)*D+x]
    __syncthreads();

    // ---- Pass 1: P_{I_i,J_j} = exp(Q K^T), written to P (HBM) ----
    // tx==0 writes; all threads compute dot so no extra sync is needed.
    for (int j_start = 0; j_start < n; j_start += BC) {
        // Cooperatively load K_{J_j}: BC rows × D cols into sK
        for (int elem = tid; elem < BC * D; elem += BR * BC)
            sK[elem] = K[j_start * D + elem];   // sK[lj*D+x] = K[(j_start+lj)*D+x]
        __syncthreads();

        for (int lj = 0; lj < BC; lj++) {
            float dot = 0.0f;
            for (int x = 0; x < D; x++)
                dot += __half2float(sQ[ty * D + x]) * __half2float(sK[lj * D + x]);
            if (tx == 0)  // one writer per (ty, lj) avoids BC duplicate writes
                P_mat[(size_t)q_idx * n + j_start + lj] = __float2half(expf(dot));
        }
        __syncthreads();
    }

    // ---- Pass 2: normalise P in HBM ----
    // Read exp scores, divide by row sum Delta_{I_i}, write normalised P back.
    if (tx == 0) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++)
            sum += __half2float(P_mat[(size_t)q_idx * n + j]);
        for (int j = 0; j < n; j++)
            P_mat[(size_t)q_idx * n + j] = __float2half(
                __half2float(P_mat[(size_t)q_idx * n + j]) / sum);
        L[q_idx] = sum;
    }
    __syncthreads();  // normalised P visible to all tx threads before pass 3

    // ---- Pass 3: stream V tiles through SRAM, read P from HBM, accumulate O ----
    float O_i[D / BC];
    for (int xi = 0; xi < D / BC; xi++) O_i[xi] = 0.0f;

    for (int j_start = 0; j_start < n; j_start += BC) {
        // Cooperatively load V_{J_j} into sV
        for (int elem = tid; elem < BC * D; elem += BR * BC)
            sV[elem] = V[j_start * D + elem];
        __syncthreads();

        for (int lj = 0; lj < BC; lj++) {
            float p = __half2float(P_mat[(size_t)q_idx * n + j_start + lj]);
            for (int xi = 0; xi < D / BC; xi++)
                O_i[xi] += p * __half2float(sV[lj * D + tx + xi * BC]);
        }
        __syncthreads();
    }

    for (int xi = 0; xi < D / BC; xi++)
        O[q_idx * D + tx + xi * BC] = __float2half(O_i[xi]);
}

// ---------------------------------------------------------------
// Flash Forward (Shared Memory = M, ref.tex block-wise)
//
// Q_{I_i}, K_{J_j}, V_{J_j} tiles all live in SRAM (main memory M).
// Tile layout identical to naive_smem_forward_kernel:
//   sQ : BR × D = 4 KB, sK : BC × D = 4 KB, sV : BC × D = 4 KB
//
// Key difference from naive: P_{I_i,J_j} = exp(Q K^T) is computed on-the-fly
// and stays in registers — it is NEVER written to HBM.
// Scale and max omitted (does not affect memory traffic analysis).
//
// Ref.tex recurrence per J-block (P_{I_i,J_j} never written to HBM):
//   Delta_i^new = Delta_i^old + rowsum(P_{I_i,J_j})
//   O_i^new     = (Delta_i^old / Delta_i^new) * O_i^old
//               + (1 / Delta_i^new) * P_{I_i,J_j} V_{J_j,:}
//
// HBM traffic per I-block (secondary memory accesses):
//   Q  : 1 × BRd   — loaded once into sQ (same as naive)
//   K  : T_c × BCd — one HBM load per J-block (same as naive)
//   V  : T_c × BCd — one HBM load per J-block (same as naive)
//   P  : 0          — P_{ij} register-resident; Delta_i in registers
//   O  : 1 × BRd   — written once at end (same as naive)
//
// Notation follows ref.tex:
//   Delta_{I_i}  — running row normaliser (sum of exp scores so far)
//   P_{I_i,J_j}  — register-resident attention weight tile
// ---------------------------------------------------------------
__global__ void flash_smem_forward_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
    half* __restrict__ O, float* __restrict__ L,
    int n
) {
    __shared__ half sQ[BR * D];
    __shared__ half sK[BC * D];
    __shared__ half sV[BC * D];

    int tx = threadIdx.x, ty = threadIdx.y;
    int q_idx = blockIdx.x * BR + ty;
    if (q_idx >= n) return;

    int tid = ty * BC + tx;

    // ---- Load Q_{I_i} into sQ once — stays in SRAM for all J-blocks ----
    int q_block_start = blockIdx.x * BR;
    for (int elem = tid; elem < BR * D; elem += BR * BC)
        sQ[elem] = Q[q_block_start * D + elem];
    __syncthreads();

    float O_i[D / BC];
    for (int xi = 0; xi < D / BC; xi++) O_i[xi] = 0.0f;
    float Delta_i = 0.0f;

    for (int j_start = 0; j_start < n; j_start += BC) {
        // Load K_{J_j} and V_{J_j} into sK and sV in the same pass.
        for (int elem = tid; elem < BC * D; elem += BR * BC) {
            sK[elem] = K[j_start * D + elem];
            sV[elem] = V[j_start * D + elem];
        }
        __syncthreads();

        // Compute P_{I_i,J_j} tile and its rowsum (ref.tex eq. P, delta).
        float P_tile[BC];
        float rowsum = 0.0f;
        for (int lj = 0; lj < BC; lj++) {
            float dot = 0.0f;
            for (int x = 0; x < D; x++)
                dot += __half2float(sQ[ty * D + x]) * __half2float(sK[lj * D + x]);
            P_tile[lj] = expf(dot);
            rowsum += P_tile[lj];
        }

        float Delta_old = Delta_i;
        Delta_i = Delta_old + rowsum;

        for (int xi = 0; xi < D / BC; xi++) {
            O_i[xi] *= Delta_old / Delta_i;
            float pv = 0.0f;
            for (int lj = 0; lj < BC; lj++)
                pv += P_tile[lj] * __half2float(sV[lj * D + tx + xi * BC]);
            O_i[xi] += pv / Delta_i;
        }
        __syncthreads();
    }

    for (int xi = 0; xi < D / BC; xi++)
        O[q_idx * D + tx + xi * BC] = __float2half(O_i[xi]);
    if (tx == 0)
        L[q_idx] = Delta_i;
}

// ---------------------------------------------------------------
// CPU reference: standard attention (scale omitted to match GPU kernels)
//   O = softmax(Q K^T) V
// Inputs/outputs are plain float arrays, row-major, shape [n, d].
// ---------------------------------------------------------------
// void cpu_attention(const float* Q, const float* K, const float* V,
//                    float* O, int n, int d) {
//     float* S = (float*)malloc((size_t)n * n * sizeof(float));  // score matrix

//     // S = Q K^T
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             float dot = 0.0f;
//             for (int x = 0; x < d; x++)
//                 dot += Q[i * d + x] * K[j * d + x];
//             S[i * n + j] = dot;
//         }
//     }

//     // Row-wise softmax on S
//     for (int i = 0; i < n; i++) {
//         float max_val = S[i * n];
//         for (int j = 1; j < n; j++)
//             if (S[i * n + j] > max_val) max_val = S[i * n + j];
//         float sum = 0.0f;
//         for (int j = 0; j < n; j++) {
//             S[i * n + j] = expf(S[i * n + j] - max_val);
//             sum += S[i * n + j];
//         }
//         for (int j = 0; j < n; j++)
//             S[i * n + j] /= sum;
//     }

//     // O = S V
//     for (int i = 0; i < n; i++)
//         for (int x = 0; x < d; x++) {
//             float acc = 0.0f;
//             for (int j = 0; j < n; j++)
//                 acc += S[i * n + j] * V[j * d + x];
//             O[i * d + x] = acc;
//         }

//     free(S);
// }

int main() {
    // seq lengths to test
    int test_sizes[] = {2048, 4096, 8192, 16384};
    // head dimension = 64
    // batch size = 1
    // number of heads = 1
    int num_tests = 4;

    printf("======================================================\n");
    printf("| %-8s | %-12s | %-12s | %-10s |\n", "Seq Len", "Naive(ms)", "Uncoal(ms)", "Flash(ms)");
    printf("======================================================\n");

    for (int t = 0; t < num_tests; t++) {
        int n = test_sizes[t];
        
        // Allocate Memory based on Current N
        size_t sz_h = (size_t)n * D * sizeof(half);
        size_t sz_p = (size_t)n * n * sizeof(half);
        
        half *d_Q, *d_K, *d_V, *d_O, *d_dO, *d_P;
        half *d_dQ, *d_dK, *d_dV;
        float *d_L, *d_m, *d_Delta;

        CHECK_CUDA(cudaMalloc(&d_Q, sz_h)); CHECK_CUDA(cudaMalloc(&d_K, sz_h)); 
        CHECK_CUDA(cudaMalloc(&d_V, sz_h)); CHECK_CUDA(cudaMalloc(&d_O, sz_h)); 
        CHECK_CUDA(cudaMalloc(&d_dO, sz_h)); CHECK_CUDA(cudaMalloc(&d_P, sz_p));
        CHECK_CUDA(cudaMalloc(&d_dQ, sz_h)); CHECK_CUDA(cudaMalloc(&d_dK, sz_h)); 
        CHECK_CUDA(cudaMalloc(&d_dV, sz_h));
        CHECK_CUDA(cudaMalloc(&d_L, n*4)); CHECK_CUDA(cudaMalloc(&d_m, n*4)); 
        CHECK_CUDA(cudaMalloc(&d_Delta, n*4));

        int flush_size = 64 * 1024 * 1024;
        float *d_flush; 
        CHECK_CUDA(cudaMalloc(&d_flush, flush_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_flush, 0, flush_size * sizeof(float)));

        // Init Data: all inputs filled with constant 0.1f for benchmarking purposes only.
        // Not realistic data — goal is to measure memory access latency, not correctness.
        // d_L and d_m (softmax statistics) are left uninitialized; only flash_optimized_half
        // uses them, and correctness is not verified here.
        float* h_dummy = (float*)malloc(n*D*4);
        for(int i=0;i<n*D;i++) h_dummy[i]=0.1f;
        float* d_temp; cudaMalloc(&d_temp, n*D*4);
        cudaMemcpy(d_temp, h_dummy, n*D*4, cudaMemcpyHostToDevice);
        float2half_kernel<<<(n*D)/256+1, 256>>>(d_Q, d_temp, n*D);
        float2half_kernel<<<(n*D)/256+1, 256>>>(d_K, d_temp, n*D);
        float2half_kernel<<<(n*D)/256+1, 256>>>(d_V, d_temp, n*D);
        float2half_kernel<<<(n*D)/256+1, 256>>>(d_O, d_temp, n*D);
        float2half_kernel<<<(n*D)/256+1, 256>>>(d_dO, d_temp, n*D);

        float scale = 1.0f / sqrtf(D);
        cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
        float ms_naive=0, ms_uncoal=0, ms_flash=0;

        compute_delta_half<<<n/256+1, 256>>>(d_dO, d_O, d_Delta, n, D);
        
        // Pre-compute P for Naive and Uncoalesced
        dim3 gridP(n/16, n/16); dim3 blockP(16, 16);
        compute_full_P_half<<<gridP, blockP>>>(d_Q, d_K, d_P, d_L, d_m, n, D, scale);

        // Regular Attention Benchmarks
        flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
        cudaEventRecord(start);
        naive_backward_kernel<<<n/256+1, 256>>>(d_Q, d_K, d_V, d_P, d_dO, d_Delta, d_dQ, d_dK, d_dV, n, D, scale);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_naive, start, stop);

        // Uncoalesced Tiled Attention Benchmarks
        dim3 block(BC, BR); dim3 grid(n / BR);
        cudaMemset(d_dQ, 0, sz_h); cudaMemset(d_dK, 0, sz_h); cudaMemset(d_dV, 0, sz_h);
        flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
        cudaEventRecord(start);
        tiled_uncoalesced_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_P, d_dO, d_dQ, d_dK, d_dV, d_Delta, scale, n);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_uncoal, start, stop);

        // Flash Attention Benchmarks
        cudaMemset(d_dQ, 0, sz_h); cudaMemset(d_dK, 0, sz_h); cudaMemset(d_dV, 0, sz_h);
        flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
        cudaEventRecord(start);
        flash_optimized_half<<<grid, block>>>(d_Q, d_K, d_V, d_dO, d_L, d_m, d_dQ, d_dK, d_dV, d_Delta, scale, n);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_flash, start, stop);

        printf("| %-8d | %-12.2f | %-12.2f | %-10.2f |\n", n, ms_naive, ms_uncoal, ms_flash);

        // Cleanup
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); 
        cudaFree(d_dO); cudaFree(d_P); cudaFree(d_dQ); cudaFree(d_dK); 
        cudaFree(d_dV); cudaFree(d_L); cudaFree(d_m); cudaFree(d_Delta);
        cudaFree(d_flush); free(h_dummy); cudaFree(d_temp);
    }
    printf("======================================================\n");

    // ---------------------------------------------------------------
    // Forward Pass Benchmarks: SRAM-as-M experiment
    //
    // Both kernels tile K and V through shared memory (L1 SRAM = M).
    // Tile size: BC × D = 16 × 64 × 2 B = 2 KB — always fits in 48 KB SRAM.
    // This directly implements the ref.tex block constraint |J|d < M/4.
    //
    // Naive  : K through sK (1 pass), raw scores → P (HBM), normalise P,
    //          V through sV, accumulate O reading P from HBM  → O(N²) HBM
    // Flash  : K through sK, V through sV, online softmax;
    //          P never written to HBM                          → O(Nd) HBM
    //
    // L2 cache is flushed between runs to avoid residual effects.
    // ---------------------------------------------------------------
    int fwd_sizes[] = {2048, 4096, 8192, 16384, 32768, 49152, 65536};
    int num_fwd = 7;

    int flush_size = 64 * 1024 * 1024;  // 256 MB — larger than L2 (72 MB)
    float *d_flush;
    CHECK_CUDA(cudaMalloc(&d_flush, flush_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_flush, 0, flush_size * sizeof(float)));

    printf("\n");
    printf("============================================================\n");
    printf("|   Forward Pass Benchmarks  |\n");
    printf("| %-8s | %-12s | %-12s | %-10s |\n",
           "Seq Len", "Naive(ms)", "Flash(ms)", "Speedup");
    printf("============================================================\n");

    for (int t = 0; t < num_fwd; t++) {
        int n = fwd_sizes[t];
        size_t sz_h = (size_t)n * D * sizeof(half);

        half *d_Q, *d_K, *d_V, *d_O_flash, *d_O_naive;
        float *d_L;

        CHECK_CUDA(cudaMalloc(&d_Q,       sz_h));
        CHECK_CUDA(cudaMalloc(&d_K,       sz_h));
        CHECK_CUDA(cudaMalloc(&d_V,       sz_h));
        CHECK_CUDA(cudaMalloc(&d_O_flash, sz_h));
        CHECK_CUDA(cudaMalloc(&d_O_naive, sz_h));
        CHECK_CUDA(cudaMalloc(&d_L, n * sizeof(float)));

        float* h_dummy = (float*)malloc((size_t)n * D * sizeof(float));
        for (int i = 0; i < n * D; i++) h_dummy[i] = 0.1f;
        float* d_temp;
        CHECK_CUDA(cudaMalloc(&d_temp, (size_t)n * D * sizeof(float)));
        cudaMemcpy(d_temp, h_dummy, (size_t)n * D * sizeof(float), cudaMemcpyHostToDevice);
        float2half_kernel<<<(n*D)/256+1, 256>>>(d_Q, d_temp, n*D);
        float2half_kernel<<<(n*D)/256+1, 256>>>(d_K, d_temp, n*D);
        float2half_kernel<<<(n*D)/256+1, 256>>>(d_V, d_temp, n*D);
        free(h_dummy); cudaFree(d_temp);

        dim3 block(BC, BR);
        dim3 grid((n + BR - 1) / BR);

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        // ==== Flash ====
        flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        flash_smem_forward_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O_flash, d_L, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_flash = 0.0f;
        cudaEventElapsedTime(&ms_flash, start, stop);

        // ==== Naive (writes P to HBM; may OOM for large N) ====
        float ms_naive = -1.0f;
        half *d_P = nullptr;
        size_t sz_p = (size_t)n * n * sizeof(half);
        if (cudaMalloc(&d_P, sz_p) == cudaSuccess) {
            flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            naive_smem_forward_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_P, d_O_naive, d_L, n);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms_naive, start, stop);
            cudaFree(d_P);
        } else {
            cudaGetLastError();
        }

        if (ms_naive >= 0.0f)
            printf("| %-8d | %-12.2f | %-12.2f | %-10.2fx |\n",
                   n, ms_naive, ms_flash, ms_naive / ms_flash);
        else
            printf("| %-8d | %-12s | %-12.2f | %-10s |\n",
                   n, "OOM", ms_flash, "N/A");

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
        cudaFree(d_O_flash); cudaFree(d_O_naive);
        cudaFree(d_L);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    printf("============================================================\n");
    cudaFree(d_flush);
    return 0;
}
