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

// Row-Wise Backward Pass
// One thread per query row i. P_{i,:} is precomputed from the forward pass.
__global__ void rowwise_backward_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const half* __restrict__ O,
    const half* __restrict__ dO,
    const half* __restrict__ P,
    half* __restrict__ dQ,
    half* __restrict__ dK,
    half* __restrict__ dV,
    float scale, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Eq (1): P_{i,:} loaded from forward pass — no recomputation needed
    float dQ_i[D];
    for (int x = 0; x < D; x++) { dQ_i[x] = 0.0f; }

    // Eq (4): gamma_i = rowsum(dO_{i,:} ⊙ O_{i,:}) — independent of j, compute once
    float gamma_i = 0.0f;
    for (int x = 0; x < D; x++) {
        gamma_i += __half2float(dO[i * D + x]) * __half2float(O[i * D + x]);
    }

    for (int j = 0; j < n; j++) {
        // Eq (1): P_{i,j} from precomputed matrix
        float P_ij = __half2float(P[i * n + j]);

        // Eq (2): dV ← dV + P_{i,:}^T dO_{i,:}
        for (int x = 0; x < D; x++) {
            atomicAddHalf(&dV[j * D + x], __float2half(P_ij * __half2float(dO[i * D + x])));
        }

        // Eq (3): dP_{i,j} = dO_{i,:} · V_{j,:}
        float dP_ij = 0.0f;
        for (int x = 0; x < D; x++) {
            dP_ij += __half2float(dO[i * D + x]) * __half2float(V[j * D + x]);
        }

        // Eq (5): dS_{i,j} = P_{i,j} ⊙ (dP_{i,j} - gamma_i)
        float dS_ij = P_ij * (dP_ij - gamma_i);

        // Eq (6): dQ_{i,:} = dS_{i,:} K  (scale from d(S_{i,j})/dQ_{i,x} = scale * K_{j,x})
        for (int x = 0; x < D; x++) {
            dQ_i[x] += dS_ij * scale * __half2float(K[j * D + x]);
        }

        // Eq (7): dK ← dK + dS_{i,:}^T Q_{i,:}
        for (int x = 0; x < D; x++) {
            atomicAddHalf(&dK[j * D + x], __float2half(dS_ij * scale * __half2float(Q[i * D + x])));
        }
    }

    // Eq (6): store dQ_{i,:}
    for (int x = 0; x < D; x++) {
        dQ[i * D + x] = __float2half(dQ_i[x]);
    }
}

// Materialized Backward Pass
// One thread per row i. Computes and writes the full dP[i,:] and dS[i,:] rows
// to global memory before using them to update dQ, dK, dV.
// This matches the slide formulation exactly:
//   dP[i,j] = dot(dO[i], V[j])                         -- stored to mat_dP
//   dS[i,j] = P[i,j] * (dP[i,j] - gamma_i)             -- stored to mat_dS
//   dQ[i]  += dS[i,j] * scale * K[j]
//   dK[j]  += dS[i,j] * scale * Q[i]                   (atomic)
//   dV[j]  += P[i,j] * dO[i]                           (atomic)
__global__ void materialized_backward_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const half* __restrict__ O,
    const half* __restrict__ dO,
    const half* __restrict__ P,
    half* __restrict__ dQ,
    half* __restrict__ dK,
    half* __restrict__ dV,
    half* __restrict__ mat_dP,   // [n x n] materialized dP
    half* __restrict__ mat_dS,   // [n x n] materialized dS
    float scale, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float gamma_i = 0.0f;
    for (int x = 0; x < D; x++)
        gamma_i += __half2float(dO[i * D + x]) * __half2float(O[i * D + x]);

    // Phase 1: materialize dP[i,:] and dS[i,:] into global memory
    for (int j = 0; j < n; j++) {
        float dP_ij = 0.0f;
        for (int x = 0; x < D; x++)
            dP_ij += __half2float(dO[i * D + x]) * __half2float(V[j * D + x]);
        mat_dP[i * n + j] = __float2half(dP_ij);

        float dS_ij = __half2float(P[i * n + j]) * (dP_ij - gamma_i);
        mat_dS[i * n + j] = __float2half(dS_ij);
    }

    // Phase 2: use the materialized rows to compute gradients
    float dQ_i[D];
    for (int x = 0; x < D; x++) dQ_i[x] = 0.0f;

    for (int j = 0; j < n; j++) {
        float P_ij  = __half2float(P[i * n + j]);
        float dS_ij = __half2float(mat_dS[i * n + j]);

        for (int x = 0; x < D; x++)
            atomicAddHalf(&dV[j * D + x], __float2half(P_ij * __half2float(dO[i * D + x])));

        for (int x = 0; x < D; x++)
            dQ_i[x] += dS_ij * scale * __half2float(K[j * D + x]);

        for (int x = 0; x < D; x++)
            atomicAddHalf(&dK[j * D + x], __float2half(dS_ij * scale * __half2float(Q[i * D + x])));
    }

    for (int x = 0; x < D; x++)
        dQ[i * D + x] = __float2half(dQ_i[x]);
}

// MatMul-Based Backward Pass (no atomics)
// One thread per row, same 1D launch as rowwise_backward_kernel.

// Step 1: thread i materializes full row i of dP.  dP[i,j] = dot(dO[i], V[j])
__global__ void compute_dP_kernel(
    const half* __restrict__ dO,
    const half* __restrict__ V,
    half* __restrict__ dP,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int j = 0; j < n; j++) {
        float acc = 0.0f;
        for (int x = 0; x < D; x++)
            acc += __half2float(dO[i * D + x]) * __half2float(V[j * D + x]);
        dP[i * n + j] = __float2half(acc);
    }
}

// Step 2: thread i materializes full row i of dS.  dS[i,j] = P[i,j] * (dP[i,j] - gamma[i])
__global__ void compute_dS_kernel(
    const half* __restrict__ P,
    const half* __restrict__ dP,
    const float* __restrict__ gamma,
    half* __restrict__ dS,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gamma[i];
    for (int j = 0; j < n; j++)
        dS[i * n + j] = __float2half(
            __half2float(P[i * n + j]) * (__half2float(dP[i * n + j]) - g)
        );
}

// Step 3: thread j owns output row j of dV.  dV[j,x] = sum_i P[i,j] * dO[i,x]  (no atomics)
__global__ void compute_dV_matmul_kernel(
    const half* __restrict__ P,
    const half* __restrict__ dO,
    half* __restrict__ dV,
    int n
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float acc[D];
    for (int x = 0; x < D; x++) acc[x] = 0.0f;
    for (int i = 0; i < n; i++) {
        float p = __half2float(P[i * n + j]);
        for (int x = 0; x < D; x++)
            acc[x] += p * __half2float(dO[i * D + x]);
    }
    for (int x = 0; x < D; x++)
        dV[j * D + x] = __float2half(acc[x]);
}

// Step 4: thread j owns output row j of dK.  dK[j,x] = scale * sum_i dS[i,j] * Q[i,x]  (no atomics)
__global__ void compute_dK_matmul_kernel(
    const half* __restrict__ dS,
    const half* __restrict__ Q,
    half* __restrict__ dK,
    float scale, int n
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float acc[D];
    for (int x = 0; x < D; x++) acc[x] = 0.0f;
    for (int i = 0; i < n; i++) {
        float ds = __half2float(dS[i * n + j]);
        for (int x = 0; x < D; x++)
            acc[x] += ds * __half2float(Q[i * D + x]);
    }
    for (int x = 0; x < D; x++)
        dK[j * D + x] = __float2half(acc[x] * scale);
}

// Step 5: thread i owns output row i of dQ.  dQ[i,x] = scale * sum_j dS[i,j] * K[j,x]  (no atomics)
__global__ void compute_dQ_matmul_kernel(
    const half* __restrict__ dS,
    const half* __restrict__ K,
    half* __restrict__ dQ,
    float scale, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc[D];
    for (int x = 0; x < D; x++) acc[x] = 0.0f;
    for (int j = 0; j < n; j++) {
        float ds = __half2float(dS[i * n + j]);
        for (int x = 0; x < D; x++)
            acc[x] += ds * __half2float(K[j * D + x]);
    }
    for (int x = 0; x < D; x++)
        dQ[i * D + x] = __float2half(acc[x] * scale);
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

// Naive Backward — No Recomputation
// One thread per query row i (same 1D launch as naive_backward_kernel).
// Thread owns row i: iterates over all j, computes dP[i,j] and dS[i,j] once,
// accumulates dQ[i] in registers (no atomics), uses atomics for dK[j] and dV[j].
__global__ void naive_nodupe_backward_kernel(
    const half* Q, const half* K, const half* V,
    const half* P,
    const half* dO, const half* O,
    half* dQ, half* dK, half* dV,
    int n, int d, float scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // query row
    if (i >= n) return;

    float delta_i = 0.0f;
    for (int x = 0; x < d; x++)
        delta_i += __half2float(dO[i * d + x]) * __half2float(O[i * d + x]);

    float dQ_i[D];
    for (int x = 0; x < D; x++) dQ_i[x] = 0.0f;

    for (int j = 0; j < n; j++) {
        float P_ij = __half2float(P[i * n + j]);

        // dV[j,x] += P[i,j] * dO[i,x]
        for (int x = 0; x < d; x++)
            atomicAddHalf(&dV[j * d + x], __float2half(P_ij * __half2float(dO[i * d + x])));

        // dP[i,j] = dot(dO[i], V[j])  — computed exactly once per (i,j)
        float dP_ij = 0.0f;
        for (int x = 0; x < d; x++)
            dP_ij += __half2float(dO[i * d + x]) * __half2float(V[j * d + x]);

        float dS_ij = P_ij * (dP_ij - delta_i) * scale;

        // dQ[i] accumulated locally — no atomics needed
        for (int x = 0; x < d; x++)
            dQ_i[x] += dS_ij * __half2float(K[j * d + x]);

        // dK[j,x] += dS[i,j] * Q[i,x]
        for (int x = 0; x < d; x++)
            atomicAddHalf(&dK[j * d + x], __float2half(dS_ij * __half2float(Q[i * d + x])));
    }

    for (int x = 0; x < d; x++)
        dQ[i * d + x] = __float2half(dQ_i[x]);
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


int main() {
    // seq lengths to test
    int test_sizes[] = {2048, 4096, 8192, 16384};
    // head dimension = 64
    // batch size = 1
    // number of heads = 1
    int num_tests = 4;

    printf("==================================================================================================================================\n");
    printf("| %-8s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s |\n", "Seq Len", "Naive(ms)", "NoDupe(ms)", "Uncoal(ms)", "Flash(ms)", "Rowwise(ms)", "KVwise(ms)", "Matrl(ms)", "MatMul(ms)");
    printf("==================================================================================================================================\n");

    for (int t = 0; t < num_tests; t++) {
        int n = test_sizes[t];
        
        // Allocate Memory based on Current N
        size_t sz_h = (size_t)n * D * sizeof(half);
        size_t sz_p = (size_t)n * n * sizeof(half);
        
        half *d_Q, *d_K, *d_V, *d_O, *d_dO, *d_P;
        half *d_dQ, *d_dK, *d_dV;
        half *d_mat_dP, *d_mat_dS;
        float *d_L, *d_m, *d_Delta;

        CHECK_CUDA(cudaMalloc(&d_Q, sz_h)); CHECK_CUDA(cudaMalloc(&d_K, sz_h));
        CHECK_CUDA(cudaMalloc(&d_V, sz_h)); CHECK_CUDA(cudaMalloc(&d_O, sz_h));
        CHECK_CUDA(cudaMalloc(&d_dO, sz_h)); CHECK_CUDA(cudaMalloc(&d_P, sz_p));
        CHECK_CUDA(cudaMalloc(&d_dQ, sz_h)); CHECK_CUDA(cudaMalloc(&d_dK, sz_h));
        CHECK_CUDA(cudaMalloc(&d_dV, sz_h));
        CHECK_CUDA(cudaMalloc(&d_mat_dP, sz_p)); CHECK_CUDA(cudaMalloc(&d_mat_dS, sz_p));
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
        float ms_naive=0, ms_uncoal=0, ms_flash=0, ms_rowwise=0, ms_kvwise=0, ms_nodupe=0, ms_matrl=0, ms_matmul=0;

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

        // Row-Wise Backward Benchmarks
        cudaMemset(d_dQ, 0, sz_h); cudaMemset(d_dK, 0, sz_h); cudaMemset(d_dV, 0, sz_h);
        flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
        cudaEventRecord(start);
        rowwise_backward_kernel<<<n/256+1, 256>>>(d_Q, d_K, d_V, d_O, d_dO, d_P, d_dQ, d_dK, d_dV, scale, n);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_rowwise, start, stop);

        // No-Dupe Backward Benchmarks (2D grid, dP/dS computed once per (i,j))
        cudaMemset(d_dQ, 0, sz_h); cudaMemset(d_dK, 0, sz_h); cudaMemset(d_dV, 0, sz_h);
        flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
        cudaEventRecord(start);
        naive_nodupe_backward_kernel<<<n/256+1, 256>>>(d_Q, d_K, d_V, d_P, d_dO, d_O, d_dQ, d_dK, d_dV, n, D, scale);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_nodupe, start, stop);

        // Materialized Backward Benchmarks (row-by-row, writes dP and dS to global memory)
        cudaMemset(d_dQ, 0, sz_h); cudaMemset(d_dK, 0, sz_h); cudaMemset(d_dV, 0, sz_h);
        flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
        cudaEventRecord(start);
        materialized_backward_kernel<<<n/256+1, 256>>>(d_Q, d_K, d_V, d_O, d_dO, d_P, d_dQ, d_dK, d_dV, d_mat_dP, d_mat_dS, scale, n);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_matrl, start, stop);

        // MatMul-Based Backward Benchmarks (separate kernels, no atomics)
        // dP = dO @ V^T, dS = elementwise, dV = P^T @ dO, dK = dS^T @ Q, dQ = dS @ K
        cudaMemset(d_dQ, 0, sz_h); cudaMemset(d_dK, 0, sz_h); cudaMemset(d_dV, 0, sz_h);
        flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
        cudaEventRecord(start);
        compute_dP_kernel<<<n/256+1, 256>>>(d_dO, d_V, d_mat_dP, n);
        compute_dS_kernel<<<n/256+1, 256>>>(d_P, d_mat_dP, d_Delta, d_mat_dS, n);
        compute_dV_matmul_kernel<<<n/256+1, 256>>>(d_P, d_dO, d_dV, n);
        compute_dK_matmul_kernel<<<n/256+1, 256>>>(d_mat_dS, d_Q, d_dK, scale, n);
        compute_dQ_matmul_kernel<<<n/256+1, 256>>>(d_mat_dS, d_K, d_dQ, scale, n);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_matmul, start, stop);

        printf("| %-8d | %-12.2f | %-12.2f | %-12.2f | %-12.2f | %-12.2f | %-12.2f | %-12.2f | %-12.2f |\n", n, ms_naive, ms_nodupe, ms_uncoal, ms_flash, ms_rowwise, ms_kvwise, ms_matrl, ms_matmul);

        // Cleanup
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
        cudaFree(d_dO); cudaFree(d_P); cudaFree(d_dQ); cudaFree(d_dK);
        cudaFree(d_dV); cudaFree(d_mat_dP); cudaFree(d_mat_dS);
        cudaFree(d_L); cudaFree(d_m); cudaFree(d_Delta);
        cudaFree(d_flush); free(h_dummy); cudaFree(d_temp);
    }
    printf("=================================================================================================================\n");
    return 0;
}