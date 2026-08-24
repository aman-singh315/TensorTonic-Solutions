#include <cuda_runtime.h>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int stride_row = blockDim.y * gridDim.y;
    int stride_col = blockDim.x * gridDim.x;
    
    for(int i = row; i < M; i += stride_row){
        for(int j = col; j < N; j+= stride_col){
            int index = i * N + j; // flatten 2-d index into 1-d memory
            
            C[index] = A[index] + B[index];
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_add_kernel<<<blocks, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}
