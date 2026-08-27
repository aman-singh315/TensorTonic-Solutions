#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_sq_sum(const float* input, float* sumv, int N) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    extern __shared__ float shared[];
    float* sqr_sum = shared;

    //sqr sum
    float local_sum = 0.0f;
    for(int j = tid; j < N; j += blockDim.x){
        local_sum += input[j] * input[j];
    }
    sqr_sum[tid] = local_sum;
    __syncthreads();

    // reduction
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride){
            sqr_sum[tid] += sqr_sum[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0){
        *sumv = sqr_sum[0];
    }
}

__global__ void divide_by_sqrt(const float* input, float* output, const float* sumv, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        float denomi = sqrtf(*sumv);
        output[i] = input[i] / denomi;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));
    int threads = 256;
    int size_i = threads * sizeof(float);

    reduce_sq_sum<<<1, 256, size_i>>>(input, d_sum, N);

    int blocks = (N + threads - 1) / threads;
    int size_t = threads * sizeof(float);
    divide_by_sqrt<<<blocks, threads, size_t>>>(input, output, d_sum, N);

    cudaDeviceSynchronize();
    cudaFree(d_sum);
}
