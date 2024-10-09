// edge_detector.cu

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

extern "C" {

// Sobel算子核函数
__constant__ float d_SobelGx[9];
__constant__ float d_SobelGy[9];

// 计算梯度大小和方向的核函数
__global__ void sobelFilterKernel(const float* input, float* gradient_magnitude, float* gradient_direction, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        float Gx = 0.0f;
        float Gy = 0.0f;

        // 应用Sobel算子
        for (int ky = -1; ky <= 1; ++ky)
        {
            for (int kx = -1; kx <= 1; ++kx)
            {
                float pixel = input[(y + ky) * width + (x + kx)];
                Gx += pixel * d_SobelGx[(ky + 1) * 3 + (kx + 1)];
                Gy += pixel * d_SobelGy[(ky + 1) * 3 + (kx + 1)];
            }
        }

        float magnitude = sqrtf(Gx * Gx + Gy * Gy);
        float direction = atan2f(Gy, Gx);

        gradient_magnitude[y * width + x] = magnitude;
        gradient_direction[y * width + x] = direction;
    }
    else if (x < width && y < height)
    {
        gradient_magnitude[y * width + x] = 0.0f;
        gradient_direction[y * width + x] = 0.0f;
    }
}

// 非极大值抑制核函数
__global__ void nonMaximumSuppressionKernel(const float* gradient_magnitude, const float* gradient_direction, float* nonMaxSuppressed, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        float angle = gradient_direction[y * width + x];
        float mag = gradient_magnitude[y * width + x];

        // 将角度量化为四个方向
        float angle_deg = angle * 180.0f / M_PI;  // 转换为度数
        if (angle_deg < 0)
            angle_deg += 180.0f;

        int direction = 0;  // 方向索引（0, 1, 2, 3）
        if ((angle_deg >= 0 && angle_deg < 22.5) || (angle_deg >= 157.5 && angle_deg <= 180))
            direction = 0;  // 0度
        else if (angle_deg >= 22.5 && angle_deg < 67.5)
            direction = 1;  // 45度
        else if (angle_deg >= 67.5 && angle_deg < 112.5)
            direction = 2;  // 90度
        else if (angle_deg >= 112.5 && angle_deg < 157.5)
            direction = 3;  // 135度

        float mag1 = 0.0f;
        float mag2 = 0.0f;

        // 与梯度方向上的邻居比较
        if (direction == 0) // 0度
        {
            mag1 = gradient_magnitude[y * width + (x - 1)];
            mag2 = gradient_magnitude[y * width + (x + 1)];
        }
        else if (direction == 1) // 45度
        {
            mag1 = gradient_magnitude[(y - 1) * width + (x + 1)];
            mag2 = gradient_magnitude[(y + 1) * width + (x - 1)];
        }
        else if (direction == 2) // 90度
        {
            mag1 = gradient_magnitude[(y - 1) * width + x];
            mag2 = gradient_magnitude[(y + 1) * width + x];
        }
        else if (direction == 3) // 135度
        {
            mag1 = gradient_magnitude[(y - 1) * width + (x - 1)];
            mag2 = gradient_magnitude[(y + 1) * width + (x + 1)];
        }

        if (mag >= mag1 && mag >= mag2)
            nonMaxSuppressed[y * width + x] = mag;
        else
            nonMaxSuppressed[y * width + x] = 0.0f;
    }
    else if (x < width && y < height)
    {
        nonMaxSuppressed[y * width + x] = 0.0f;
    }
}

// 双阈值处理核函数
__global__ void doubleThresholdKernel(float* nonMaxSuppressed, unsigned char* edgeMap, float highThreshold, float lowThreshold, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float val = nonMaxSuppressed[y * width + x];

        if (val >= highThreshold)
            edgeMap[y * width + x] = 2; // 强边缘（新发现）
        else if (val >= lowThreshold)
            edgeMap[y * width + x] = 1; // 弱边缘
        else
            edgeMap[y * width + x] = 0; // 非边缘
    }
}

// 边缘连接（滞后阈值）核函数（改进版）
__global__ void edgeTrackingHysteresisKernel(unsigned char* edgeMap, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * width;

    if (idx < width * height)
    {
        if (edgeMap[idx] == 2) // 强边缘
        {
            bool is_edge = true;
            while (is_edge)
            {
                is_edge = false;

                int x = idx % width;
                int y = idx / width;

                // 检查8邻域
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        int nx = x + dx;
                        int ny = y + dy;

                        if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                        {
                            int nidx = ny * width + nx;
                            if (edgeMap[nidx] == 1) // 弱边缘
                            {
                                edgeMap[nidx] = 2; // 升级为强边缘
                                is_edge = true; // 继续检查
                            }
                        }
                    }
                }
            }
        }
    }
}

// 将边缘图转换为0或255
__global__ void thresholdEdgeMapKernel(unsigned char* edgeMap, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * width;

    if (idx < width * height)
    {
        if (edgeMap[idx] == 2)
            edgeMap[idx] = 255; // 强边缘
        else
            edgeMap[idx] = 0;   // 非边缘或弱边缘
    }
}

// 主机函数，供Python调用
__host__ void edge_detector(const float* input_image, unsigned char* output_edge_map, int width, int height)
{
    // Sobel算子
    float h_SobelGx[9] = {-1, 0, 1,
                          -2, 0, 2,
                          -1, 0, 1};
    float h_SobelGy[9] = {1, 2, 1,
                          0, 0, 0,
                          -1, -2, -1};

    // 将Sobel算子复制到常量内存
    cudaMemcpyToSymbol(d_SobelGx, h_SobelGx, 9 * sizeof(float));
    cudaMemcpyToSymbol(d_SobelGy, h_SobelGy, 9 * sizeof(float));

    // 分配设备内存
    float* d_input;
    float* d_gradient_magnitude;
    float* d_gradient_direction;
    float* d_nonMaxSuppressed;
    unsigned char* d_edgeMap;

    size_t imageSizeFloat = width * height * sizeof(float);
    size_t imageSizeChar = width * height * sizeof(unsigned char);

    cudaMalloc((void**)&d_input, imageSizeFloat);
    cudaMalloc((void**)&d_gradient_magnitude, imageSizeFloat);
    cudaMalloc((void**)&d_gradient_direction, imageSizeFloat);
    cudaMalloc((void**)&d_nonMaxSuppressed, imageSizeFloat);
    cudaMalloc((void**)&d_edgeMap, imageSizeChar);

    // 将输入图像复制到设备
    cudaMemcpy(d_input, input_image, imageSizeFloat, cudaMemcpyHostToDevice);

    // 设置线程块和网格尺寸
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x -1) / blockDim.x, (height + blockDim.y -1) / blockDim.y);

    // 第一步：Sobel滤波
    sobelFilterKernel<<<gridDim, blockDim>>>(d_input, d_gradient_magnitude, d_gradient_direction, width, height);

    // 第二步：非极大值抑制
    nonMaximumSuppressionKernel<<<gridDim, blockDim>>>(d_gradient_magnitude, d_gradient_direction, d_nonMaxSuppressed, width, height);

    // 第三步：计算自适应阈值
    // 将非极大值抑制后的梯度幅值复制回主机
    float* h_nonMaxSuppressed = new float[width * height];
    cudaMemcpy(h_nonMaxSuppressed, d_nonMaxSuppressed, imageSizeFloat, cudaMemcpyDeviceToHost);

    // 计算梯度幅值的直方图
    int numBins = 64;
    float maxMag = 0.0f;

    // 找到最大梯度值
    for (int i = 0; i < width * height; ++i)
    {
        if (h_nonMaxSuppressed[i] > maxMag)
            maxMag = h_nonMaxSuppressed[i];
    }

    // 初始化直方图
    int* histogram = new int[numBins]();
    float binSize = (maxMag + FLT_EPSILON) / numBins; // 避免除以零

    // 填充直方图
    for (int i = 0; i < width * height; ++i)
    {
        int bin = static_cast<int>(h_nonMaxSuppressed[i] / binSize);
        if (bin >= numBins)
            bin = numBins - 1;
        histogram[bin]++;
    }

    // 计算累积分布函数（CDF）
    int total = width * height;
    int sum = 0;
    float percentile = 0.985f; // 90%百分位
    float highThreshold = 0.0f;
    for (int i = numBins - 1; i >= 0; --i)
    {
        sum += histogram[i];
        if (sum >= total * (1.0f - percentile))
        {
            highThreshold = (i + 0.5f) * binSize; // 取bin中心值
            break;
        }
    }

    // 低阈值为高阈值的一定比例
    float lowThreshold = highThreshold * 0.8f;

    // 释放主机内存
    delete[] histogram;
    delete[] h_nonMaxSuppressed;

    // 第四步：双阈值处理
    doubleThresholdKernel<<<gridDim, blockDim>>>(d_nonMaxSuppressed, d_edgeMap, highThreshold, lowThreshold, width, height);

    // 第五步：边缘连接（滞后阈值）
    // 由于CUDA内核中无法使用递归或动态内存分配，我们采用迭代方式
    // 如果需要更精细的边缘连接，可以多次调用该内核

    // 调用边缘跟踪核函数
    edgeTrackingHysteresisKernel<<<gridDim, blockDim>>>(d_edgeMap, width, height);

    // 第六步：最终处理边缘图
    thresholdEdgeMapKernel<<<gridDim, blockDim>>>(d_edgeMap, width, height);

    // 将结果复制回主机
    cudaMemcpy(output_edge_map, d_edgeMap, imageSizeChar, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_gradient_magnitude);
    cudaFree(d_gradient_direction);
    cudaFree(d_nonMaxSuppressed);
    cudaFree(d_edgeMap);
}

}
