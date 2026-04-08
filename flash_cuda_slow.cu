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
    return 0;
}
