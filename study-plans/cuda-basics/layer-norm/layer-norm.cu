#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int stride_row = blockDim.y * gridDim.y;
    int stride_col = blockDim.x * gridDim.x;

    for(int i = row; i < M; i+= stride_row){

        // compute mean
        float sum = 0.0f;
        for(int j = 0; j < N; j++){
            int index = i *  N +j;
            sum += input[index];
        }
        float mean = sum / N;

        //compute variance
        float variance = 0.0f;
        for(int j = 0; j < N; j++){
            float diff = input[i * N + j] - mean;
            variance += diff * diff; 
        }
        variance /= N;

        float inv_std = 1.0f / sqrtf(variance + eps);
        // normalized  elements
        for(int j = col; j < N; j  += stride_col){
            int index = i * N + j;
            float normalize = (input[index] - mean) * inv_std;
            output[index] = normalize * gamma[j] + beta[j];
        }
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}
