#include <cuda_runtime.h>

__global__ void mean_variance_kernel(const float* input, float* mean_out, float* var_out, int N) {
    
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    float* sum_shared = shared;
    float* var_shared = shared + blockDim.x;

    float local_sum = 0.0f;
    // mean calculation
    for(int j = tid; j < N; j+= blockDim.x){
        local_sum += input[j];
    }
    sum_shared[tid] = local_sum;
    __syncthreads();

    //parallel reduction
    for(int stride = blockDim.x/2; stride > 0; stride /= 2){
        if(tid < stride){
            sum_shared[tid] += sum_shared[tid + stride];            
        }
        __syncthreads();
    }

    float mean = sum_shared[0] / N;
    if(tid == 0){
        mean_out[0] = mean;
    }

    __syncthreads();


    // compute variance
    float local_var = 0.0f;
    for(int j = tid; j < N; j += blockDim.x){
        float x = input[j];
        float diff = x - mean;
        local_var += diff * diff; 
    }
    var_shared[tid] = local_var;
    __syncthreads();

    // parallel reduction
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride){
            var_shared[tid] += var_shared[tid + stride];        
        }
        __syncthreads();
    }
    if(tid == 0){
        var_out[0] = var_shared[0] / N;        
    }
    
}

extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t shared_mem_size = 2 * threads * sizeof(float);
    cudaMemset(mean_out, 0, sizeof(float));
    cudaMemset(var_out, 0, sizeof(float));
    mean_variance_kernel<<<blocks, threads, shared_mem_size>>>(input, mean_out, var_out, N);
    cudaDeviceSynchronize();
}
