#include <cuda_runtime.h>

__global__ void dot_kernel(const float* A, const float* B, float* result, int N) {
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    float* sum = shared;
    float local_sum = 0.0f;

    for(int i = tid; i < N; i+= blockDim.x){
        local_sum += A[i] * B[i];
    }
    sum[tid] = local_sum;
    __syncthreads();

    //partiall reduction
    for(int stride = blockDim.x/2; stride  > 0; stride /= 2){
        if(tid < stride){
            sum[tid] += sum[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0){
        result[0] = sum[0];
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t shared_mem_size = threads * sizeof(float);
    cudaMemset(result, 0, sizeof(float));
    dot_kernel<<<blocks, threads,shared_mem_size>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
