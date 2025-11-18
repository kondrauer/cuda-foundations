#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <cstdio>

__global__ void box_blur(unsigned char* img, unsigned char* img_blur, int width, int height, int channels, int r, float* weights)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; c++) {

        float sum = 0.0f;
        float weight_sum = 0.0f;

        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx ++) {
                int sx = x + dx;
                int sy = y + dy;

                if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                    int weight_idx = (dy + r) * (2 * r + 1) + (dx + r);
                    float w = weights[weight_idx];

                    unsigned char pixel_value = img[(sy * width + sx) * channels + c];
                    sum += w * pixel_value;
                    weight_sum += w;
                }

            
            }
        }

        if (weight_sum != 0) {
            sum /= weight_sum;
        }

        img_blur[(y * width + x) * channels + c] = (unsigned char)sum;

    }

}


int main()
{
    const int radius = 8;
    const int k = 2*radius + 1;

    float* weights = new float[k * k];
    for (int i = 0; i < k * k; i++) {
        weights[i] = 1.0f;
    }

    int width, height, channels;

    // Load image as 8-bit unsinged char array
    unsigned char* img = stbi_load("image.jpg", &width, &height, &channels, 0);

    if (!img) {
        printf("Failed to load image\n");
        return 1;
    }

    printf("Loaded image with width=%d height=%d channels=%d\n", width, height, channels);

    unsigned char* d_img, *d_img_blur;
    float* d_weights;

    cudaMalloc(&d_img, sizeof(unsigned char) * width * height * channels);
    cudaMalloc(&d_img_blur, sizeof(unsigned char) * width * height * channels);
    cudaMalloc(&d_weights, sizeof(float) * k * k);
    cudaMemcpy(d_weights, weights, sizeof(float) * k * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img, img, sizeof(unsigned char) * width * height * channels, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    box_blur<<<gridSize, blockSize>>>(d_img, d_img_blur, width, height, channels, k, d_weights);

    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    cudaMemcpy(img, d_img_blur, sizeof(unsigned char) * width * height * channels, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_img_blur);
    cudaFree(d_weights);

    stbi_write_jpg("image-blur.jpg", width, height, channels, img, 100);

    stbi_image_free(img);

    return 0;
}