#include <cuda_runtime.h>
#include <float.h>

//compare two values (value, index) pairs
//larger values wins and if two same value then smallest idx wins
__device__ __forceinline__
bool better(float candidate_value, float candidate_idx, float current_value, float  current_idx){
    if(candidate_value > current_value) return true;
    if(candidate_value == current_value){
        //condidate is valid and current is invalid
        if(candidate_idx >= 0 && current_idx < 0) return true;

        // if both valid smaller indx wins
        if(candidate_idx >= 0 && current_idx >= 0 && candidate_idx < current_idx){
            return true;
        }
    }
    return false;
}

__global__ void argmax_kernel(const float* input, float* block_vals, int* block_idxs, int N) {


    extern __shared__ unsigned char shared_mem[];
    
    float* s_vals = reinterpret_cast<float*>(shared_mem);
    int* s_idxs = reinterpret_cast<int*>(shared_mem + blockDim.x * sizeof(float));
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    //bound check
    if(idx < N){
        s_vals[tid] = input[idx];
        s_idxs[tid] = idx; 
    }else{
        s_vals[tid] = -INFINITY;
        s_idxs[tid] = -1;
    }
    __syncthreads();

    //parallel reduction
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride){
            int other = tid + stride;
            if(better(
                s_vals[other],
                s_idxs[other],
                s_vals[tid],
                s_idxs[tid]
            )){
                s_vals[tid] = s_vals[other];
                s_idxs[tid] = s_idxs[other];
            }
        }
        __syncthreads();
    }

    // write the block winner
    if(tid == 0){
        block_vals[blockIdx.x] = s_vals[0];
        block_idxs[blockIdx.x] = s_idxs[0];
    }
}

__global__ void argmax_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {

    extern __shared__ unsigned char shared_mem[];

    float* s_vals = reinterpret_cast<float*>(shared_mem);
    int* s_idxs = reinterpret_cast<int*>(shared_mem + blockDim.x * sizeof(float));

    int tid = threadIdx.x;

    float max_val = -INFINITY;
    int max_idx = -1;


    //cooperative grid-stride loop over output
    for(int i = tid; i < num_blocks; i += blockDim.x){
        float candidate_value = block_vals[i];
        float candidate_idx = block_idxs[i];

        if(better(
            candidate_value,
            candidate_idx,
            max_val,
            max_idx
        )){
            max_val = candidate_value;
            max_idx = candidate_idx;
        }
    }
    s_vals[tid] = max_val;
    s_idxs[tid] = max_idx;
    __syncthreads();

    // block-tree reduction
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride){
            int other = tid + stride;
            if(better(
                s_vals[other],
                s_idxs[other],
                s_vals[tid],
                s_idxs[tid]
            )){
                s_vals[tid] = s_vals[other];
                s_idxs[tid] = s_idxs[other];
            }
        }
        __syncthreads();
    }

    // getting the armax index
    if(tid == 0){
        *result = s_idxs[0];
    }
}

extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals = nullptr;
    int* block_idxs = nullptr;
    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    size_t shared_bytes = threads *  sizeof(float) + threads * sizeof(int);
    argmax_kernel<<<blocks, threads, shared_bytes>>>(input, block_vals, block_idxs, N);
    argmax_finalize_kernel<<<1, threads, shared_bytes>>>(block_vals, block_idxs, result, blocks);
    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}
