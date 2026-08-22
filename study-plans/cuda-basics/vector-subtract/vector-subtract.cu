#include <cuda_runtime.h>

__global__ void vector_sub(const float* A, const float* B, float* C, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < N; i += stride){
        C[i] = A[i] - B[i];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_sub<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}