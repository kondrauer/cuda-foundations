#include <cstdio>

__global__ void VecAdd(float *A, float *B, float *C, int* N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < *N) {
        C[i] = A[i] + B[i];
    }
    
}

int main()
{
    const int host_N = 3;

    float host_A[host_N] = {1, 2, 3};
    float host_B[host_N] = {1, 2, 3};
    float host_C[host_N];

    int *device_N;
    float *device_A, *device_B, *device_C;

    cudaMalloc((void**)&device_N, sizeof(int));
    cudaMalloc((void**)&device_A, host_N * sizeof(float));
    cudaMalloc((void**)&device_B, host_N * sizeof(float));
    cudaMalloc((void**)&device_C, host_N * sizeof(float));

    cudaMemcpy(device_N, &host_N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_A, host_A, host_N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, host_N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_C, host_C, host_N * sizeof(float), cudaMemcpyHostToDevice);

    VecAdd<<<1, host_N>>>(device_A, device_B, device_C, device_N);

    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    cudaMemcpy(host_A, device_A, host_N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_B, device_B, host_N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_C, device_C, host_N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_N);
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    for (int i = 0; i < host_N; i++)
    {
        printf("%f ", host_C[i]);
    }

    return 0;
}