#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, int N, int kN) {
    // extern __shared__ float shared[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int outN = N - kN + 1;

    if(i < outN){
        float sum = 0.0f;

        for(int a = 0; a < kN; a++){
            sum += input[i + a] * kernel[a];
        }

        output[i] = sum;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int N, int kN) {
    int outN = N - kN + 1;
    int threads = 256;
    dim3 blocks((outN + 255) / 256);
    conv1d_kernel<<<blocks, threads>>>(input, kernel, output, N, kN);
    cudaDeviceSynchronize();
}
