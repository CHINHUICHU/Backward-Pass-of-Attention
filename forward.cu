#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define D 64         // Head Dimension
#define BR 16        // Block Row Size
#define BC 16        // Block Col Size

#define CHECK_CUDA(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("\n[CUDA ERROR] %s:%d: %s (code: %d)\n", __FILE__, __LINE__, cudaGetErrorString(error), error); \
        exit(1); \
    } \
}

// Resets the "sticky" error state so one OOM/crash doesn't kill the whole program
inline void clear_cuda_error() {
    cudaGetLastError();
}

// ---------------------------------------------------------------
// Initialization & Utility Kernels
// ---------------------------------------------------------------

// Sets half values directly on GPU to avoid CPU-GPU transfer overhead
__global__ void fill_half_kernel(half* data, float value, size_t size) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) data[i] = __float2half(value);
}

__global__ void flush_l2_cache_kernel(float* buffer, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) buffer[idx] += 1.0f;
}

// ---------------------------------------------------------------
// Row-Wise Forward Pass (Naive)
// ---------------------------------------------------------------
__global__ void rowwise_forward_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ S,
    half* __restrict__ O,
    int T, int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T) return;

    size_t row_offset_S = (size_t)i * T;
    size_t row_offset_QD = (size_t)i * d;

    // Eq 1: S_{i,:} = Q_{i,:} K^T
    for (int j = 0; j < T; j++) {
        float dot = 0.0f;
        size_t row_offset_KD = (size_t)j * d;
        for (int x = 0; x < d; x++) {
            dot += __half2float(Q[row_offset_QD + x]) * __half2float(K[row_offset_KD + x]);
        }
        S[row_offset_S + j] = dot;
    }

    // Eq 2: SoftMax P directly use S space
    float sum = 0.0f;
    for (int j = 0; j < T; j++) { 
        S[row_offset_S + j] = expf(S[row_offset_S + j]); 
        sum += S[row_offset_S + j]; 
    }
    float inv_sum = (sum > 0) ? (1.0f / sum) : 0.0f;
    for (int j = 0; j < T; j++) {
        S[row_offset_S + j] *= inv_sum;
    }

    // Eq 3: O_{i,:} = P_{i,:} V
    for (int x = 0; x < d; x++) {
        float acc = 0.0f;
        for (int j = 0; j < T; j++) {
            acc += S[row_offset_S + j] * __half2float(V[(size_t)j * d + x]);
        }
        O[row_offset_QD + x] = __float2half(acc);
    }
}

// ---------------------------------------------------------------
// Row-Wise Forward Pass (Separate Kernels)
// ---------------------------------------------------------------

// Step 1: thread i computes row i of S = Q K^T
__global__ void compute_S_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    float* __restrict__ S,
    int T, int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T) return;
    size_t row_Q = (size_t)i * d;
    for (int j = 0; j < T; j++) {
        float dot = 0.0f;
        size_t row_K = (size_t)j * d;
        for (int x = 0; x < d; x++)
            dot += __half2float(Q[row_Q + x]) * __half2float(K[row_K + x]);
        S[(size_t)i * T + j] = dot;
    }
}

// Step 2: thread i applies softmax to row i of S in-place
__global__ void compute_softmax_kernel(
    float* __restrict__ S,
    int T
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T) return;
    size_t row = (size_t)i * T;
    float sum = 0.0f;
    for (int j = 0; j < T; j++) { S[row + j] = expf(S[row + j]); sum += S[row + j]; }
    float inv_sum = (sum > 0) ? (1.0f / sum) : 0.0f;
    for (int j = 0; j < T; j++) S[row + j] *= inv_sum;
}

// Step 3: thread i computes row i of O = P V
__global__ void compute_O_kernel(
    const float* __restrict__ P,
    const half* __restrict__ V,
    half* __restrict__ O,
    int T, int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T) return;
    size_t row_P = (size_t)i * T;
    size_t row_O = (size_t)i * d;
    for (int x = 0; x < d; x++) {
        float acc = 0.0f;
        for (int j = 0; j < T; j++)
            acc += P[row_P + j] * __half2float(V[(size_t)j * d + x]);
        O[row_O + x] = __float2half(acc);
    }
}

// ---------------------------------------------------------------
// Flash Forward (Memory Efficient)
// ---------------------------------------------------------------
__global__ void flash_forward_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float* __restrict__ Delta,
    int T
) {
    int tx = threadIdx.x;   
    int ty = threadIdx.y;   
    int q_idx = blockIdx.x * BR + ty;
    if (q_idx >= T) return;

    size_t q_row_offset = (size_t)q_idx * D;

    float O_i[D / BC];
    for (int k = 0; k < D / BC; k++) {
        O_i[k] = 0.0f;
    }
    float Delta_i = 0.0f;

    for (int j_start = 0; j_start < T; j_start += BC) {
        float P_tile[BC];
        for (int k_row = 0; k_row < BC; k_row++) {
            int global_k = j_start + k_row;
            if (global_k < T) {
                float dot = 0.0f;
                size_t k_row_offset = (size_t)global_k * D;
                for (int x = 0; x < D; x++) {
                    dot += __half2float(Q[q_row_offset + x]) * __half2float(K[k_row_offset + x]);
                }
                P_tile[k_row] = expf(dot);
            } else {
                P_tile[k_row] = 0.0f;
            }
        }

        for (int k_row = 0; k_row < BC; k_row++) {
            int global_k = j_start + k_row;
            if (global_k >= T) continue;

            float Delta_old = Delta_i;
            Delta_i += P_tile[k_row];
            float inv_Di    = (Delta_i > 0) ? (1.0f / Delta_i) : 0.0f;

            size_t v_row_offset = (size_t)global_k * D;
            for (int k = 0; k < D / BC; k++) {
                int x = tx + k * BC;
                O_i[k] = (Delta_old * inv_Di) * O_i[k] + (P_tile[k_row] * inv_Di) * __half2float(V[v_row_offset + x]);
            }
        }
    }

    for (int k = 0; k < D / BC; k++) {
        O[q_row_offset + tx + k * BC] = __float2half(O_i[k]);
    }
    
    if (tx == 0) {
        Delta[q_idx] = Delta_i;
    }
}

int main() {
    int fwd_sizes[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    int num_fwd = 8;

    int flush_size = 64 * 1024 * 1024;
    float *d_flush;
    CHECK_CUDA(cudaMalloc(&d_flush, flush_size * sizeof(float)));

    printf("=======================================================================\n");
    printf("|                   Forward Pass Benchmark                          |\n");
    printf("| %-8s | %-15s | %-15s | %-15s |\n", "Seq Len", "Naive(ms)", "Separate(ms)", "Flash(ms)");
    printf("=======================================================================\n");

    for (int t = 0; t < num_fwd; t++) {
        int T = fwd_sizes[t];
        size_t sz_h = (size_t)T * D * sizeof(half);
        size_t sz_s = (size_t)T * T * sizeof(float);

        half *d_Q, *d_K, *d_V, *d_O;
        float *d_Delta, *d_S = nullptr;

        CHECK_CUDA(cudaMalloc(&d_Q, sz_h));
        CHECK_CUDA(cudaMalloc(&d_K, sz_h));
        CHECK_CUDA(cudaMalloc(&d_V, sz_h));
        CHECK_CUDA(cudaMalloc(&d_O, sz_h));
        CHECK_CUDA(cudaMalloc(&d_Delta, T * sizeof(float)));

        // Attempt allocation for Naive scratchpad (S). Will fail if > GPU VRAM.
        cudaError_t s_malloc_err = cudaMalloc(&d_S, sz_s);
        if (s_malloc_err != cudaSuccess) {
            clear_cuda_error(); // Resets flag so we don't skip the Flash kernel too
            d_S = nullptr;
        }

        // --- INITIALIZATION ---
        size_t total_elements = (size_t)T * D;
        fill_half_kernel<<<(total_elements + 255)/256, 256>>>(d_Q, 0.1f, total_elements);
        fill_half_kernel<<<(total_elements + 255)/256, 256>>>(d_K, 0.1f, total_elements);
        fill_half_kernel<<<(total_elements + 255)/256, 256>>>(d_V, 0.1f, total_elements);
        cudaMemset(d_O, 0, sz_h);
        cudaMemset(d_Delta, 0, T * sizeof(float));

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        // --- NAIVE RUN ---
        float ms_naive = -1.0f;
        if (d_S != nullptr) {
            flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            rowwise_forward_kernel<<<(T+255)/256, 256>>>(d_Q, d_K, d_V, d_S, d_O, T, D);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms_naive, start, stop);
        }

        // --- SEPARATE KERNELS RUN ---
        float ms_sep = -1.0f;
        if (d_S != nullptr) {
            cudaMemset(d_O, 0, sz_h);
            flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            compute_S_kernel<<<(T+255)/256, 256>>>(d_Q, d_K, d_S, T, D);
            compute_softmax_kernel<<<(T+255)/256, 256>>>(d_S, T);
            compute_O_kernel<<<(T+255)/256, 256>>>(d_S, d_V, d_O, T, D);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms_sep, start, stop);
        }

        // --- FLASH RUN ---
        float ms_ff = -1.0f;
        flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        {
            dim3 block(BC, BR);
            dim3 grid((T + BR - 1) / BR);
            flash_forward_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, d_Delta, T);
        }
        cudaEventRecord(stop);
        const char *flash_err_str = nullptr;
        {
            cudaError_t sync_err = cudaEventSynchronize(stop);
            if (sync_err == cudaSuccess) {
                cudaEventElapsedTime(&ms_ff, start, stop);
            } else {
                flash_err_str = cudaGetErrorString(sync_err);
                clear_cuda_error();
            }
        }

        // --- LOGGING ---
        char n_str[16], s_str[16], f_str[64];
        if (ms_naive < 0) sprintf(n_str, "OOM/SKIP"); else sprintf(n_str, "%.2f", ms_naive);
        if (ms_sep < 0)   sprintf(s_str, "OOM/SKIP"); else sprintf(s_str, "%.2f", ms_sep);
        if (ms_ff < 0)    snprintf(f_str, sizeof(f_str), "ERR: %s", flash_err_str ? flash_err_str : "unknown");
        else              sprintf(f_str, "%.2f", ms_ff);
        printf("| %-8d | %-15s | %-15s | %-15s |\n", T, n_str, s_str, f_str);

        if (d_S) cudaFree(d_S);
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_Delta);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    printf("=======================================================================\n");

    cudaFree(d_flush);
    return 0;
}