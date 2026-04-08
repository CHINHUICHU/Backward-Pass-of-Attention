#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    printf("=== GPU SPECIFICATIONS: %s ===\n", props.name);
    printf("Compute Capability:       %d.%d\n", props.major, props.minor);
    printf("Global Memory (VRAM):     %.2f GB\n", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    printf("L2 Cache Size:            %.2f MB (%d bytes)\n", 
           props.l2CacheSize / (1024.0 * 1024.0), props.l2CacheSize);
    
    printf("Maximum shared memory available to a single thread block:     %.2f KB\n", props.sharedMemPerBlock / 1024.0);
    printf("Registers per Block:      %d\n", props.regsPerBlock);
    
    // Theoretical limit check
    // We used N=16384, D=64
    // Size of one matrix (Q, K, or V) = 16384 * 64 * 2 = 2,097,152 bytes (2 MB)
    printf("\n=== ANALYSIS FOR N=16384 ===\n");
    double tensor_size_mb = (16384.0 * 64.0 * 2.0) / (1024.0 * 1024.0);
    printf("Size of ONE tensor (Q):   %.2f MB\n", tensor_size_mb);
    printf("Min Working Set (Q+K+V):  %.2f MB\n", tensor_size_mb * 3);
    
    if ((tensor_size_mb * 3) > (props.l2CacheSize / (1024.0 * 1024.0))) {
        printf("RESULT: working set > L2 Cache -> CACHE THRASHING CONFIRMED.\n");
    } else {
        printf("RESULT: Fits in L2 Cache.\n");
    }

    return 0;
}
