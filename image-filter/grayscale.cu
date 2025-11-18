#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <cstdio>

__global__ void grayscale(unsigned char* img, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = (y * width + x) * channels;
    unsigned char r = img[index + 0];
    unsigned char g = img[index + 1];
    unsigned char b = img[index + 2];

    unsigned char gray = 0.299*r + 0.587*g + 0.114*b;
    img[index + 0] = gray;
    img[index + 1] = gray;
    img[index + 2] = gray;
}


int main()
{
    int width, height, channels;

    // Load image as 8-bit unsinged char array
    unsigned char* img = stbi_load("image.jpg", &width, &height, &channels, 0);

    
    if (!img) {
        printf("Failed to load image\n");
        return 1;
    }

    printf("Loaded image with width=%d height=%d channels=%d\n", width, height, channels);

    unsigned char* d_img;

    cudaMalloc(&d_img, sizeof(unsigned char) * width * height* channels);
    cudaMemcpy(d_img, img, sizeof(unsigned char) * width * height* channels, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    grayscale<<<gridSize, blockSize>>>(d_img, width, height, channels);

    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    cudaMemcpy(img, d_img, sizeof(unsigned char) * width * height* channels, cudaMemcpyDeviceToHost);

    cudaFree(d_img);

    

    stbi_write_jpg("image-grayscale.jpg", width, height, channels, img, 100);

    stbi_image_free(img);

    return 0;
}