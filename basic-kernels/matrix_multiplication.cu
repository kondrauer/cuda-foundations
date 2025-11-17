#include <cstdio>

__global__ void MatMul2D(float* A, float* B, float* C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // A = MxK, B = KxN, C = MxN
    // M = number of rows in A and C
    // N = number of columns in B and C
    // K = number of columns in A and rows in B
    if(row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main()
{
    int M = 2, K = 3, N = 1;

    float A[M * K] = {
        1, 2, 3,
        4, 5, 6,
    };
    float B[K * N] = {
        9,
        8,
        7
    };
    float C[M * N] = {0};

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    MatMul2D<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // result should be C = [ 46, 118 ]
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}