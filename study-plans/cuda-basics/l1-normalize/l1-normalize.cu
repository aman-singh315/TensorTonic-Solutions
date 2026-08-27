#include <cuda_runtime.h>
#include <math.h>

__global__ void l1_normalize_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    extern __shared__ float shared[];
    float* sum_shared = shared;

    float local_sum = 0.0f;
    // running sum
    for(int j = tid; j < N; j+= blockDim.x){
        local_sum += fabsf(input[j]); 
    }
    sum_shared[tid] = local_sum;
    __syncthreads();

    //partial reduction
    for(int stride = blockDim.x/2; stride > 0; stride /=2){
        if(tid < stride){
            sum_shared[tid] += sum_shared[tid + stride];
        }
        __syncthreads();
    }

    //normalize
    if(i < N){
        output[i] = input[i] / sum_shared[0];
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int size_t = threads * sizeof(float);
    l1_normalize_kernel<<<blocks, threads, size_t>>>(input, output, N);
    cudaDeviceSynchronize();
}
