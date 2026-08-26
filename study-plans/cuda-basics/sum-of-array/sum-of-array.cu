#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int st = blockDim.x * gridDim.x;

    extern __shared__ float shared[];

    float* sum = shared;
    float local_sum = 0.0f;

    for(int i = idx; i < N; i+= st){
        local_sum += input[i];
    }
    sum[tid] = local_sum;
    __syncthreads();

    // paratial reduction
    for(int stride = blockDim.x /2; stride > 0; stride /= 2){
        if(tid < stride){
            sum[tid] += sum[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0){
        atomicAdd(result, sum[0]);
    }
    
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int size_t = threads * sizeof(float);
    cudaMemset(result, 0, sizeof(float));
    sum_kernel<<<blocks, threads, size_t>>>(input, result, N);
    cudaDeviceSynchronize();
}
