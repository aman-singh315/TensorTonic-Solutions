#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int stride_row = blockDim.y * gridDim.y;
    int stride_col = blockDim.x * gridDim.x;

    for(int i = row; i < M; i+= stride_row){

        // calculate input square 
        float sum_sq = 0.0f;
        for(int k = 0; k < N; k++){
            float x = input[i * N + k];
            sum_sq += x * x;
        }
        
        float rms = sqrtf(sum_sq / N + eps);
        for(int j = col; j < N; j+= stride_col){
            int index = i * N + j;
            output[index] = (input[index] / rms) * gamma[j];
        }
    }
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    rms_norm_kernel<<<blocks, threads>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}