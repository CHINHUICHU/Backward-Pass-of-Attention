#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define D 64         // Head Dimension
#define BR 64        // Block Row Size
#define BC 32        // Block Col Size

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

__global__ void fill_float_kernel(float* data, float value, size_t size) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) data[i] = value;
}

__global__ void flush_l2_cache_kernel(float* buffer, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) buffer[idx] += 1.0f;
}

// ---------------------------------------------------------------
// General MatMul Kernel
// ---------------------------------------------------------------

__device__ inline float to_float(float x) { return x; }

template<typename T> __device__ inline T from_float(float x);
template<> __device__ inline float from_float<float>(float x) { return x; }

// C[M x N] = A[M x K] @ B_eff[K x N], one thread per row of C.
// transB=true: B is stored as [N x K], i.e. B[j, x] = B[j*K + x].
template<typename TA, typename TB, typename TC>
__global__ void matmul_kernel(
    const TA* __restrict__ A,
    const TB* __restrict__ B,
    TC* __restrict__ C,
    int M, int N, int K,
    bool transB
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    for (int j = 0; j < N; j++) {
        float acc = 0.0f;
        for (int x = 0; x < K; x++) {
            float a = to_float(A[(size_t)i * K + x]);
            float b = transB ? to_float(B[(size_t)j * K + x])
                             : to_float(B[(size_t)x * N + j]);
            acc += a * b;
        }
        C[(size_t)i * N + j] = from_float<TC>(acc);
    }
}

// ---------------------------------------------------------------
// Standard Attention (register-only, no shared memory)
// ---------------------------------------------------------------

// In-place softmax over rows of S[T x T].
// 1D block (BR threads), thread tx owns one row. Two passes over global S.
__global__ void compute_softmax_kernel(float* __restrict__ S, int T) {
    int row = blockIdx.x * BR + threadIdx.x;
    if (row >= T) return;

    float row_sum = 0.0f;
    for (int j = 0; j < T; j++) {
        float e = expf(S[(size_t)row * T + j]);
        S[(size_t)row * T + j] = e;
        row_sum += e;
    }
    float inv_sum = 1.0f / row_sum;
    for (int j = 0; j < T; j++) S[(size_t)row * T + j] *= inv_sum;
}

// Runs standard attention: S = Q K^T, P = softmax(S), O = P V.
// Reuses matmul_kernel for both matmuls. S is a [T x T] float scratch buffer.
void standard_attention(
    const float* Q, const float* K, const float* V,
    float* S, float* O,
    int T, int d
) {
    int Tr = (T + BR - 1) / BR;

    // S = Q @ K^T
    matmul_kernel<float, float, float><<<Tr, BR>>>(Q, K, S, T, T, D, true);

    // softmax(S) in-place
    compute_softmax_kernel<<<Tr, BR>>>(S, T);

    // O = S @ V
    matmul_kernel<float, float, float><<<Tr, BR>>>(S, V, O, T, D, T, false);
}

// ---------------------------------------------------------------
// Flash Forward (Register-only)
// ---------------------------------------------------------------
//
// Qi[D], Oi[D], Pij[BC] held entirely in per-thread registers.
// K and V read directly from HBM each j-iteration.
// Threads are fully independent: no shared memory, no __syncthreads.
// Launch: grid = (Tr,), block = (Br,). Thread tx owns row tx.
__global__ void flash_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ Delta,
    int T
) {
    int q_idx = blockIdx.x * BR + threadIdx.x;
    if (q_idx >= T) return;

    float qi[D] = {}, oi[D] = {};
    for (int x = 0; x < D; x++) qi[x] = Q[(size_t)q_idx * D + x];

    float Delta_i = 0.0f;
    int Tc = (T + BC - 1) / BC;

    for (int j = 0; j < Tc; j++) {
        float pij[BC] = {};
        float Delta_old = Delta_i;

        // Pij = exp(Qi . Kj^T); load Kj row-by-row from HBM
        for (int y = 0; y < BC; y++) {
            int k_idx = j * BC + y;
            float dot = 0.0f;
            if (k_idx < T) {
                for (int x = 0; x < D; x++)
                    dot += qi[x] * K[(size_t)k_idx * D + x];
            }
            pij[y] = expf(dot);
            Delta_i += pij[y];
        }

        // Oi <- (Delta_old/Delta_i)*Oi + (1/Delta_i)*Pij*Vj; load Vj from HBM
        float inv_Di    = 1.0f / Delta_i;
        float scale_old = Delta_old * inv_Di;
        for (int x = 0; x < D; x++) {
            float pv = 0.0f;
            for (int y = 0; y < BC; y++) {
                int k_idx = j * BC + y;
                if (k_idx < T) pv += pij[y] * V[(size_t)k_idx * D + x];
            }
            oi[x] = scale_old * oi[x] + inv_Di * pv;
        }
    }

    for (int x = 0; x < D; x++) O[(size_t)q_idx * D + x] = oi[x];
    Delta[q_idx] = Delta_i;
}

int main() {
    int fwd_sizes[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536};
    int num_fwd = 7;

    int flush_size = 64 * 1024 * 1024;
    float *flush_buf;
    CHECK_CUDA(cudaMalloc(&flush_buf, flush_size * sizeof(float)));

    printf("===========================================================\n");
    printf("|              Forward Pass Benchmark                    |\n");
    printf("| %-8s | %-15s | %-15s |\n", "Seq Len", "Separate(ms)", "Flash(ms)");
    printf("===========================================================\n");

    for (int t = 0; t < num_fwd; t++) {
        int T = fwd_sizes[t];
        size_t sz_f = (size_t)T * D * sizeof(float);
        size_t sz_s = (size_t)T * T * sizeof(float);

        float *Q, *K, *V, *O;
        float *Delta, *S = nullptr;

        CHECK_CUDA(cudaMalloc(&Q, sz_f));
        CHECK_CUDA(cudaMalloc(&K, sz_f));
        CHECK_CUDA(cudaMalloc(&V, sz_f));
        CHECK_CUDA(cudaMalloc(&O, sz_f));
        CHECK_CUDA(cudaMalloc(&Delta, T * sizeof(float)));

        // Attempt allocation for Naive scratchpad (S). Will fail if > GPU VRAM.
        cudaError_t s_malloc_err = cudaMalloc(&S, sz_s);
        if (s_malloc_err != cudaSuccess) {
            clear_cuda_error(); // Resets flag so we don't skip the Flash kernel too
            S = nullptr;
        }

        // --- INITIALIZATION ---
        size_t total_elements = (size_t)T * D;
        fill_float_kernel<<<(total_elements + 255)/256, 256>>>(Q, 0.1f, total_elements);
        fill_float_kernel<<<(total_elements + 255)/256, 256>>>(K, 0.1f, total_elements);
        fill_float_kernel<<<(total_elements + 255)/256, 256>>>(V, 0.1f, total_elements);
        cudaMemset(O, 0, sz_f);
        cudaMemset(Delta, 0, T * sizeof(float));

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        // --- SEPARATE KERNELS RUN ---
        float ms_sep = -1.0f;
        if (S != nullptr) {
            cudaMemset(O, 0, sz_f);
            flush_l2_cache_kernel<<<flush_size/256, 256>>>(flush_buf, flush_size);
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            standard_attention(Q, K, V, S, O, T, D);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms_sep, start, stop);
        }

        // --- FLASH RUN ---
        float ms_ff = -1.0f;
        flush_l2_cache_kernel<<<flush_size/256, 256>>>(flush_buf, flush_size);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        {
            dim3 block(BR);
            dim3 grid((T + BR - 1) / BR);
            flash_forward_kernel<<<grid, block>>>(Q, K, V, O, Delta, T);
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
        char s_str[16], f_str[64];
        if (ms_sep < 0)   sprintf(s_str, "OOM/SKIP"); else sprintf(s_str, "%.2f", ms_sep);
        if (ms_ff < 0)    snprintf(f_str, sizeof(f_str), "ERR: %s", flash_err_str ? flash_err_str : "unknown");
        else              sprintf(f_str, "%.2f", ms_ff);
        printf("| %-8d | %-15s | %-15s |\n", T, s_str, f_str);

        if (S) cudaFree(S);
        cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O); cudaFree(Delta);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    printf("===========================================================\n");

    cudaFree(flush_buf);
    return 0;
}