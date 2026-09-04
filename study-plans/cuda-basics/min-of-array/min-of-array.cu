#include <cuda_runtime.h>
#include <float.h>

__global__ void init_result(float* result) {
    result[0] = FLT_MAX;
}

__global__ void min_kernel(const float* input, float* result, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shared[];
    
    float local_min = FLT_MAX;
    //finding min 
    for(int i = idx; i < N; i += blockDim.x){
        local_min = min(local_min , input[i]);
    }
    shared[tid] = local_min;
    __syncthreads();

    //partial reduction
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride){
            shared[tid] = min(shared[tid] , shared[tid + stride]);
        }
        __syncthreads();
    }
    if(tid == 0){
        result[0] = shared[0];
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int size_t = threads * sizeof(float);
    init_result<<<1, 1>>>(result);
    min_kernel<<<blocks, threads, size_t>>>(input, result, N);
    cudaDeviceSynchronize();
}
