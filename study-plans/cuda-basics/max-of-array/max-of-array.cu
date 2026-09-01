#include <cuda_runtime.h>
#include <float.h>

__global__ void max_kernel(const float* input, float* result, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    extern __shared__ float shared[];
    float* max_vec = shared;

    //finding partial max
    float local_max = -FLT_MAX;
    for(int i = idx; i < N; i += blockDim.x){
        local_max = max(local_max , input[i]);
    }
    max_vec[tid] = local_max;
    __syncthreads();

    // partial reduction
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2 ){
        if(tid < stride){
            max_vec[tid] = max(max_vec[tid], max_vec[tid + stride]);
        }
        __syncthreads();
    }
    if(tid == 0){
        result[tid] = max_vec[0];
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float neg_inf = -FLT_MAX;
    int size_st = threads * sizeof(float);
    cudaMemcpy(result, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    max_kernel<<<blocks, threads, size_st>>>(input, result, N);
    cudaDeviceSynchronize();
}
