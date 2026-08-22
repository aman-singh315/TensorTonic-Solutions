#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // find maximum
    float max_val = input[0];
    for(int j = 1; j < N; j++){
        max_val = fmaxf(max_val, input[j]);
    }

    // compute sum of exponentials
    float sum = 0.0f;
    for(int k = 0; k < N; k++){
        sum += __expf(input[k] - max_val);
    }

    // compute softmax
    for(int i = 0; i < N; i++){
        output[i] = __expf(input[i] - max_val) / sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}