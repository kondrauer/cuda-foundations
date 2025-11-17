#include <cstdio>

__global__ void print_indices_kernel()
{
    // this is actually run  and buffered on the cuda device and flushed after calling cudaDeviceSynchronize()
    // VERY SLOW, ONLY FOR DEBUGGING PURPOSES
    printf("block (%d, %d, %d) thread (%d,%d,%d) \n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);

    int tid_in_block = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    long global_tid = (long)block_id * threads_per_block + tid_in_block;

    printf("tid_in_block=%d block_id=%d global_tid=%ld\n", tid_in_block, block_id, global_tid);
}

int main()
{
    dim3 blocks(2, 2, 1);
    dim3 threads(4, 2, 1);

    print_indices_kernel<<<blocks, threads>>>();

    cudaDeviceSynchronize();
}