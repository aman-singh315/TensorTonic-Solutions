#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < N; i+= stride){
        output[i] = input[i] / (1.0f + __expf(-input[i]));        
    }
    
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    swish_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
