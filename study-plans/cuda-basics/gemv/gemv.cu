#include <cuda_runtime.h>

__global__ void gemv_kernel(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M){
        float sum = 0.0f;
        for(int j = 0; j < N; j++){
            sum += A[row * N + j] * x[j];
        }
        y[row] = sum;
    }

    
}

extern "C" void solve(const float* A, const float* x, float* y, int M, int N) {
    dim3 threads(256);
    dim3 blocks((M + 255) / 256);
    gemv_kernel<<<blocks, threads>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}
