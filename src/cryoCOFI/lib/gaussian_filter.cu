// gaussian_filter.cu

#include <cuda_runtime.h>
#include <math.h>

extern "C" {

// CUDA核函数，执行高斯滤波
__global__ void gaussianFilterKernel(const float* input, float* output, int width, int height,
                                     const float* kernel, int kernel_radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float sum = 0.0f;

    for (int ky = -kernel_radius; ky <= kernel_radius; ++ky)
    {
        for (int kx = -kernel_radius; kx <= kernel_radius; ++kx)
        {
            int nx = x + kx;
            int ny = y + ky;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                float pixel = input[ny * width + nx];
                float weight = kernel[(ky + kernel_radius) * (2 * kernel_radius + 1) + (kx + kernel_radius)];
                sum += pixel * weight;
            }
        }
    }

    output[y * width + x] = sum;
}

// 主机函数，供Python调用
__host__ void gaussian_filter(const float* input, float* output, int width, int height, int kernel_size)
{
    int kernel_radius = kernel_size / 2;
    int kernel_length = kernel_size * kernel_size;

    // 生成高斯核
    float sigma = kernel_size / 6.0f;  // 常用的经验值
    float* h_kernel = new float[kernel_length];

    float sum = 0.0f;
    for (int y = -kernel_radius; y <= kernel_radius; ++y)
    {
        for (int x = -kernel_radius; x <= kernel_radius; ++x)
        {
            float exponent = -(x * x + y * y) / (2 * sigma * sigma);
            float value = expf(exponent);
            h_kernel[(y + kernel_radius) * kernel_size + (x + kernel_radius)] = value;
            sum += value;
        }
    }

    // 归一化高斯核
    for (int i = 0; i < kernel_length; ++i)
    {
        h_kernel[i] /= sum;
    }

    // 分配设备内存
    float* d_input;
    float* d_output;
    float* d_kernel;
    size_t image_size = width * height * sizeof(float);
    size_t kernel_size_bytes = kernel_length * sizeof(float);

    cudaMalloc((void**)&d_input, image_size);
    cudaMalloc((void**)&d_output, image_size);
    cudaMalloc((void**)&d_kernel, kernel_size_bytes);

    // 将数据复制到设备
    cudaMemcpy(d_input, input, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size_bytes, cudaMemcpyHostToDevice);

    // 设置线程块和网格尺寸
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // 启动CUDA核函数
    gaussianFilterKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height,
                                                d_kernel, kernel_radius);

    // 将结果复制回主机
    cudaMemcpy(output, d_output, image_size, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    // 释放主机内存
    delete[] h_kernel;
}

}
