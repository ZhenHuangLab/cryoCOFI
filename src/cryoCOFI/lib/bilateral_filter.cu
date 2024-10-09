// bilateral_filter.cu

#include <cuda_runtime.h>
#include <math.h>

extern "C" {

// CUDA核函数，执行双边滤波
__global__ void bilateralFilterKernel(const float* input, float* output, int width, int height,
                                      int kernel_radius, float sigma_color, float sigma_space)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float sum = 0.0f;
    float norm_factor = 0.0f;
    float center_val = input[y * width + x];

    for (int dy = -kernel_radius; dy <= kernel_radius; ++dy)
    {
        for (int dx = -kernel_radius; dx <= kernel_radius; ++dx)
        {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                float neighbor_val = input[ny * width + nx];

                float spatial_dist = dx * dx + dy * dy;
                float spatial_weight = expf(-spatial_dist / (2.0f * sigma_space * sigma_space));

                float color_dist = neighbor_val - center_val;
                float color_weight = expf(-color_dist * color_dist / (2.0f * sigma_color * sigma_color));

                float weight = spatial_weight * color_weight;

                sum += neighbor_val * weight;
                norm_factor += weight;
            }
        }
    }

    output[y * width + x] = sum / norm_factor;
}

// 主机函数，供Python调用
__host__ void bilateral_filter(const float* input, float* output, int width, int height,
                               int kernel_radius, float sigma_color, float sigma_space)
{
    // 分配设备内存
    float* d_input;
    float* d_output;
    size_t size = width * height * sizeof(float);
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // 将输入数据复制到设备
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // 设置线程块和网格尺寸
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // 启动CUDA核函数
    bilateralFilterKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height,
                                                 kernel_radius, sigma_color, sigma_space);

    // 将结果复制回主机
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_output);
}

}
