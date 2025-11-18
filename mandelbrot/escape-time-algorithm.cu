#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../image-filter/stb_image_write.h"
#include <math.h>
#include <vector>

__device__ inline unsigned char lerp_uc(unsigned char a, unsigned char b, float t) {
    return (unsigned char)(a + (b - a) * t + 0.5f);
}

__global__ void escape_time_algorithm(unsigned char* img, int width, int height, int channels, unsigned char* d_palette, int palette_size)
{
    int x_0 = blockIdx.x * blockDim.x + threadIdx.x;
    int y_0 = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_0 >= width || y_0 >= height) return;

    // seahorse valley: center_x = -0.75f; center_y = 0.1f; zoom = 50.0f;
    // elephant valley: center_x = 0.25f; center_y = 0.0f; zoom = 100.0f;
    // fractal spirals: center_x = -1.25066f; center_y = 0.02012f; zoom = 10000.0f;
    // "default" view: center_x = -0.5f; center_y = 0.0f; zoom = 0.6f;
    float center_x = -1.25066f;
    float center_y = 0.02012f;
    float zoom = 10000.0f;
    float aspect_ratio = 1.0f;

    float half_width = 1.0f / zoom;
    float half_height = half_width / aspect_ratio;

    float x_min = center_x - half_width;
    float x_max = center_x + half_width;
    float y_min = center_y - half_height;
    float y_max = center_y + half_height;

    float x_scaled = x_min + (x_max - x_min) * (x_0 / float(width - 1));
    float y_scaled = y_min + (y_max - y_min) * (y_0 / float(height - 1));

    float x = 0;
    float y = 0;
    int i = 0;
    int max_iter = 5000;

    float x_squared = x * x;
    float y_squared = y * y;
    
    while (x_squared + y_squared <= 4 && i < max_iter) {
        float x_temp = x_squared - y_squared + x_scaled;
        y = 2 * x * y + y_scaled;
        x = x_temp;

        i++;

        x_squared = x * x;
        y_squared = y * y;
    }

    int p_index = (y_0 * width + x_0) * channels;

    if (i >= max_iter) {
        img[p_index + 0] = 0;
        img[p_index + 1] = 0;
        img[p_index + 2] = 0;
    } else {
        // Enhanced smooth coloring algorithm
        float log_zn = logf(x_squared + y_squared) / 2.0f;
        float nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
        float smooth_i = float(i) + 1.0f - nu;
        
        float t = smooth_i / float(max_iter);
        t = sqrtf(t);  // Apply square root for more contrast at boundaries
        float fidx = t * float(palette_size - 1);
        int idx0 = int(floorf(fidx));
        int idx1 = idx0 + 1;
        if (idx1 >= palette_size) idx1 = palette_size - 1;

        float localT = fidx - float(idx0);

        int base0 = idx0 * 3;
        int base1 = idx1 * 3;
        unsigned char r0 = d_palette[base0 + 0];
        unsigned char g0 = d_palette[base0 + 1];
        unsigned char b0 = d_palette[base0 + 2];
        unsigned char r1 = d_palette[base1 + 0];
        unsigned char g1 = d_palette[base1 + 1];
        unsigned char b1 = d_palette[base1 + 2];

        unsigned char r = lerp_uc(r0, r1, localT);
        unsigned char g = lerp_uc(g0, g1, localT);
        unsigned char b = lerp_uc(b0, b1, localT);

        img[p_index + 0] = r;
        img[p_index + 1] = g;
        img[p_index + 2] = b;
    }
}


int main()
{

    const int width = 4096;
    const int height = 4096;
    const int channels = 3;
    const int palette_size = 256;

    unsigned char *d_img, *d_palette = nullptr;

    std::vector<unsigned char> img(width * height * channels);
    std::vector<unsigned char> palette(palette_size * 3);

    for (int k = 0; k < palette_size; ++k) {
        float t = float(k) / float(palette_size - 1);
        
        // Create a blue-to-gold gradient with some variation
        float r, g, b;
        
        if (t < 0.1f) {
            // Very deep blue to blue
            float local_t = t / 0.1f;
            r = 0.0f + 0.05f * local_t;
            g = 0.05f + 0.15f * local_t;
            b = 0.3f + 0.5f * local_t;
        } else if (t < 0.3f) {
            // Blue to bright cyan
            float local_t = (t - 0.1f) / 0.2f;
            r = 0.05f + 0.15f * local_t;
            g = 0.2f + 0.7f * local_t;
            b = 0.8f + 0.2f * local_t;
        } else if (t < 0.5f) {
            // Cyan to bright yellow
            float local_t = (t - 0.3f) / 0.2f;
            r = 0.2f + 0.8f * local_t;
            g = 0.9f + 0.1f * local_t;
            b = 1.0f - 0.9f * local_t;
        } else if (t < 0.7f) {
            // Yellow to bright orange
            float local_t = (t - 0.5f) / 0.2f;
            r = 1.0f;
            g = 1.0f - 0.2f * local_t;
            b = 0.1f - 0.1f * local_t;
        } else if (t < 0.85f) {
            // Orange to red-orange
            float local_t = (t - 0.7f) / 0.15f;
            r = 1.0f;
            g = 0.8f - 0.4f * local_t;
            b = 0.0f + 0.2f * local_t;
        } else {
            // Red-orange to bright white
            float local_t = (t - 0.85f) / 0.15f;
            r = 1.0f;
            g = 0.4f + 0.6f * local_t;
            b = 0.2f + 0.8f * local_t;
        }
        
        palette[3*k + 0] = (unsigned char)(r * 255.0f);
        palette[3*k + 1] = (unsigned char)(g * 255.0f);
        palette[3*k + 2] = (unsigned char)(b * 255.0f);
    }

    cudaMalloc(&d_img, sizeof(unsigned char) * width * height * channels);
    cudaMalloc(&d_palette, sizeof(unsigned char) * palette_size * channels);

    cudaMemcpy(d_palette, palette.data(), sizeof(unsigned char) * palette_size * channels, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    escape_time_algorithm<<<gridSize, blockSize>>>(d_img, width, height, channels, d_palette, palette_size);

    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    cudaMemcpy(img.data(), d_img, sizeof(unsigned char) * width * height * channels, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_palette);

    stbi_write_jpg("mandebrot-spirals.jpg", width, height, channels, img.data(), 100);

    return 0;
}