// 文件名：canny.cu

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

extern "C" {

// CUDA 错误检查宏
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error: %s (err_num=%d)\n",               \
                    cudaGetErrorString(err), err);                         \
            exit(err);                                                     \
        }                                                                  \
    } while (0)

// 生成高斯核函数
void generate_gaussian_kernel(float sigma, float** kernel, int* kernel_radius)
{
    int radius = (int)ceil(3 * sigma);
    int size = 2 * radius + 1;
    float* h_kernel = (float*)malloc(size * size * sizeof(float));
    float sum = 0.0f;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            float value = expf(-(x*x + y*y)/(2 * sigma * sigma));
            h_kernel[(y + radius) * size + (x + radius)] = value;
            sum += value;
        }
    }
    // 归一化核
    for (int i = 0; i < size * size; ++i) {
        h_kernel[i] /= sum;
    }
    *kernel = h_kernel;
    *kernel_radius = radius;
}

// 高斯滤波核函数（使用共享内存）
__global__ void gaussian_filter_kernel(const float* __restrict__ input, float* __restrict__ output, int width, int height, const float* __restrict__ kernel, int kernel_radius)
{
    extern __shared__ float shared_mem[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int shared_width = blockDim.x + 2 * kernel_radius;

    int shared_x = threadIdx.x + kernel_radius;
    int shared_y = threadIdx.y + kernel_radius;

    // 将数据加载到共享内存
    if (x < width && y < height) {
        shared_mem[shared_y * shared_width + shared_x] = input[y * width + x];
    } else {
        shared_mem[shared_y * shared_width + shared_x] = 0.0f;
    }

    // 处理边缘
    for (int i = 0; i < kernel_radius; ++i) {
        // 左边界
        if (threadIdx.x < kernel_radius) {
            int left_x = x - kernel_radius + i;
            int shared_left_x = threadIdx.x + i;
            if (left_x >= 0 && y < height) {
                shared_mem[shared_y * shared_width + shared_left_x] = input[y * width + left_x];
            } else {
                shared_mem[shared_y * shared_width + shared_left_x] = 0.0f;
            }
        }
        // 右边界
        if (threadIdx.x >= blockDim.x - kernel_radius) {
            int right_x = x + kernel_radius - (blockDim.x - threadIdx.x - 1 - i);
            int shared_right_x = threadIdx.x + 2 * kernel_radius - i;
            if (right_x < width && y < height) {
                shared_mem[shared_y * shared_width + shared_right_x] = input[y * width + right_x];
            } else {
                shared_mem[shared_y * shared_width + shared_right_x] = 0.0f;
            }
        }
        // 上边界
        if (threadIdx.y < kernel_radius) {
            int top_y = y - kernel_radius + i;
            int shared_top_y = threadIdx.y + i;
            if (top_y >= 0 && x < width) {
                shared_mem[shared_top_y * shared_width + shared_x] = input[top_y * width + x];
            } else {
                shared_mem[shared_top_y * shared_width + shared_x] = 0.0f;
            }
        }
        // 下边界
        if (threadIdx.y >= blockDim.y - kernel_radius) {
            int bottom_y = y + kernel_radius - (blockDim.y - threadIdx.y - 1 - i);
            int shared_bottom_y = threadIdx.y + 2 * kernel_radius - i;
            if (bottom_y < height && x < width) {
                shared_mem[shared_bottom_y * shared_width + shared_x] = input[bottom_y * width + x];
            } else {
                shared_mem[shared_bottom_y * shared_width + shared_x] = 0.0f;
            }
        }
    }

    // 同步线程，确保共享内存已加载完成
    __syncthreads();

    // 执行卷积
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int kernel_size = 2 * kernel_radius + 1;
    for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
        for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
            float pixel = shared_mem[(shared_y + ky) * shared_width + (shared_x + kx)];
            float weight = kernel[(ky + kernel_radius) * kernel_size + (kx + kernel_radius)];
            sum += pixel * weight;
        }
    }
    output[y * width + x] = sum;
}

// // Sobel 梯度计算核函数（使用共享内存）
// __global__ void sobel_filter_kernel(const float* __restrict__ input, float* __restrict__ gradient, float* __restrict__ direction, int width, int height)
// {
//     __shared__ float shared_mem[18][18]; // 假设 blockDim.x = 16, kernel_radius = 1
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     int shared_x = threadIdx.x + 1;
//     int shared_y = threadIdx.y + 1;

//     // 将数据加载到共享内存
//     if (x < width && y < height) {
//         shared_mem[shared_y][shared_x] = input[y * width + x];
//     } else {
//         shared_mem[shared_y][shared_x] = 0.0f;
//     }

//     // 处理边缘
//     if (threadIdx.x == 0) {
//         // 左边界
//         int left_x = x - 1;
//         if (left_x >= 0 && y < height) {
//             shared_mem[shared_y][shared_x - 1] = input[y * width + left_x];
//         } else {
//             shared_mem[shared_y][shared_x - 1] = 0.0f;
//         }
//     }
//     if (threadIdx.x == blockDim.x - 1) {
//         // 右边界
//         int right_x = x + 1;
//         if (right_x < width && y < height) {
//             shared_mem[shared_y][shared_x + 1] = input[y * width + right_x];
//         } else {
//             shared_mem[shared_y][shared_x + 1] = 0.0f;
//         }
//     }
//     if (threadIdx.y == 0) {
//         // 上边界
//         int top_y = y - 1;
//         if (top_y >= 0 && x < width) {
//             shared_mem[shared_y - 1][shared_x] = input[top_y * width + x];
//         } else {
//             shared_mem[shared_y - 1][shared_x] = 0.0f;
//         }
//     }
//     if (threadIdx.y == blockDim.y - 1) {
//         // 下边界
//         int bottom_y = y + 1;
//         if (bottom_y < height && x < width) {
//             shared_mem[shared_y + 1][shared_x] = input[bottom_y * width + x];
//         } else {
//             shared_mem[shared_y + 1][shared_x] = 0.0f;
//         }
//     }

//     // 处理四个角
//     if (threadIdx.x == 0 && threadIdx.y == 0) {
//         // 左上角
//         int left_x = x - 1;
//         int top_y = y - 1;
//         if (left_x >= 0 && top_y >= 0) {
//             shared_mem[shared_y - 1][shared_x - 1] = input[top_y * width + left_x];
//         } else {
//             shared_mem[shared_y - 1][shared_x - 1] = 0.0f;
//         }
//     }
//     if (threadIdx.x == blockDim.x -1 && threadIdx.y == 0) {
//         // 右上角
//         int right_x = x + 1;
//         int top_y = y -1;
//         if (right_x < width && top_y >= 0) {
//             shared_mem[shared_y - 1][shared_x +1] = input[top_y * width + right_x];
//         } else {
//             shared_mem[shared_y - 1][shared_x +1] = 0.0f;
//         }
//     }
//     if (threadIdx.x == 0 && threadIdx.y == blockDim.y -1) {
//         // 左下角
//         int left_x = x -1;
//         int bottom_y = y +1;
//         if (left_x >= 0 && bottom_y < height) {
//             shared_mem[shared_y +1][shared_x -1] = input[bottom_y * width + left_x];
//         } else {
//             shared_mem[shared_y +1][shared_x -1] = 0.0f;
//         }
//     }
//     if (threadIdx.x == blockDim.x -1 && threadIdx.y == blockDim.y -1) {
//         // 右下角
//         int right_x = x +1;
//         int bottom_y = y +1;
//         if (right_x < width && bottom_y < height) {
//             shared_mem[shared_y +1][shared_x +1] = input[bottom_y * width + right_x];
//         } else {
//             shared_mem[shared_y +1][shared_x +1] = 0.0f;
//         }
//     }

//     // 同步线程，确保共享内存已加载完成
//     __syncthreads();

//     // 执行梯度计算
//     if (x >= width || y >= height) return;

//     float Gx = 0.0f;
//     float Gy = 0.0f;

//     // Sobel 核
//     const float sobel_x[3][3] = {
//         {-1, 0, 1},
//         {-2, 0, 2},
//         {-1, 0, 1}
//     };
//     const float sobel_y[3][3] = {
//         {-1, -2, -1},
//         {0,   0,  0},
//         {1,   2,  1}
//     };

//     for (int ky = -1; ky <= 1; ++ky) {
//         for (int kx = -1; kx <=1; ++kx) {
//             float pixel = shared_mem[shared_y + ky][shared_x + kx];
//             Gx += pixel * sobel_x[ky + 1][kx + 1];
//             Gy += pixel * sobel_y[ky + 1][kx + 1];
//         }
//     }

//     gradient[y * width + x] = sqrtf(Gx * Gx + Gy * Gy);
//     direction[y * width + x] = atan2f(Gy, Gx);
// }

__global__ void sobel_filter_kernel(const float* __restrict__ input, float* __restrict__ gradient, float* __restrict__ direction, int width, int height)
{
    __shared__ float shared_mem[18][18]; // 假设 blockDim.x = 16, kernel_radius = 1
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int shared_x = threadIdx.x + 1;
    int shared_y = threadIdx.y + 1;

    // 初始化共享内存元素
    shared_mem[shared_y][shared_x] = 0.0f;

    // 将数据加载到共享内存
    if (x < width && y < height) {
        shared_mem[shared_y][shared_x] = input[y * width + x];
    } else {
        shared_mem[shared_y][shared_x] = 0.0f;
    }

    // 边界处理
    if (threadIdx.x == 0) {
        // 左边界
        if (x > 0 && y < height) {
            shared_mem[shared_y][shared_x - 1] = input[y * width + x - 1];
        } else {
            shared_mem[shared_y][shared_x - 1] = 0.0f;
        }
    }
    if (threadIdx.x == blockDim.x - 1) {
        // 右边界
        if (x + 1 < width && y < height) {
            shared_mem[shared_y][shared_x + 1] = input[y * width + x + 1];
        } else {
            shared_mem[shared_y][shared_x + 1] = 0.0f;
        }
    }
    if (threadIdx.y == 0) {
        // 上边界
        if (y > 0 && x < width) {
            shared_mem[shared_y - 1][shared_x] = input[(y - 1) * width + x];
        } else {
            shared_mem[shared_y - 1][shared_x] = 0.0f;
        }
    }
    if (threadIdx.y == blockDim.y - 1) {
        // 下边界
        if (y + 1 < height && x < width) {
            shared_mem[shared_y + 1][shared_x] = input[(y + 1) * width + x];
        } else {
            shared_mem[shared_y + 1][shared_x] = 0.0f;
        }
    }

    __syncthreads();

    if (x >= width || y >= height) return;

    float Gx = 0.0f;
    float Gy = 0.0f;

    // Sobel 核
    const float sobel_x[3][3] = {
        { -1.0f, 0.0f, 1.0f },
        { -2.0f, 0.0f, 2.0f },
        { -1.0f, 0.0f, 1.0f }
    };
    const float sobel_y[3][3] = {
        { -1.0f, -2.0f, -1.0f },
        {  0.0f,  0.0f,  0.0f },
        {  1.0f,  2.0f,  1.0f }
    };

    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            float pixel = shared_mem[shared_y + ky][shared_x + kx];
            Gx += pixel * sobel_x[ky + 1][kx + 1];
            Gy += pixel * sobel_y[ky + 1][kx + 1];
        }
    }

    float magnitude = sqrtf(Gx * Gx + Gy * Gy);
    float angle = atan2f(Gy, Gx);

    // 检查 NaN 和 Inf
    if (isnan(magnitude) || isinf(magnitude)) {
        magnitude = 0.0f;
    }
    if (isnan(angle) || isinf(angle)) {
        angle = 0.0f;
    }

    gradient[y * width + x] = magnitude;
    direction[y * width + x] = angle;
}

// 计算梯度幅值的最小值和最大值的核函数
__global__ void min_max_kernel(const float* __restrict__ gradient, float* min_vals, float* max_vals, int width, int height)
{
    __shared__ float s_min[256];
    __shared__ float s_max[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total = width * height;

    float val = (idx < total) ? gradient[idx] : FLT_MAX;
    float val_max = (idx < total) ? gradient[idx] : -FLT_MAX;

    // 初始化共享内存
    s_min[tid] = val;
    s_max[tid] = val_max;

    __syncthreads();

    // 归约最小值和最大值
    for (int s = blockDim.x / 2; s > 0; s >>=1) {
        if (tid < s) {
            if (s_min[tid + s] < s_min[tid]) {
                s_min[tid] = s_min[tid + s];
            }
            if (s_max[tid + s] > s_max[tid]) {
                s_max[tid] = s_max[tid + s];
            }
        }
        __syncthreads();
    }

    // 将每个块的最小值和最大值写入全局内存
    if (tid == 0) {
        min_vals[blockIdx.x] = s_min[0];
        max_vals[blockIdx.x] = s_max[0];
    }
}

// 计算直方图的核函数
__global__ void histogram_kernel(const float* __restrict__ gradient, int* hist, int width, int height, float min_val, float max_val, int hist_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;

    if (idx >= total) return;

    float val = gradient[idx];
    int bin = (int)((val - min_val) / (max_val - min_val + 1e-6f) * hist_size);
    if (bin >= hist_size) bin = hist_size - 1;
    if (bin < 0) bin = 0;

    atomicAdd(&hist[bin], 1);
}

// 非极大值抑制核函数
__global__ void non_max_suppression_kernel(const float* gradient, const float* direction, float* nms_output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 修改边界检查，允许处理更多像素
    if (x >= width || y >= height) return;

    float angle = direction[y * width + x];
    float current = gradient[y * width + x];

    // 将角度转换到 [0, 180) 度
    angle = fmodf(angle + M_PI, M_PI) * 180.0f / M_PI;

    float neighbor1 = 0.0f;
    float neighbor2 = 0.0f;

    // 初始化输出
    nms_output[y * width + x] = 0.0f;

    // 只在非边界像素进行插值比较
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // 根据梯度方向，选择要比较的邻居
        if ((angle >= 157.5f) || (angle < 22.5f)) {
            // 0度方向（水平）
            neighbor1 = gradient[y * width + (x + 1)];
            neighbor2 = gradient[y * width + (x - 1)];
        }
        else if (angle >= 22.5f && angle < 67.5f) {
            // 45度方向
            neighbor1 = gradient[(y + 1) * width + (x - 1)];
            neighbor2 = gradient[(y - 1) * width + (x + 1)];
        }
        else if (angle >= 67.5f && angle < 112.5f) {
            // 90度方向（垂直）
            neighbor1 = gradient[(y + 1) * width + x];
            neighbor2 = gradient[(y - 1) * width + x];
        }
        else if (angle >= 112.5f && angle < 157.5f) {
            // 135度方向
            neighbor1 = gradient[(y - 1) * width + (x - 1)];
            neighbor2 = gradient[(y + 1) * width + (x + 1)];
        }

        // 非极大值抑制
        if (current > neighbor1 && current > neighbor2) {
            nms_output[y * width + x] = current;
        }
    } else {
        // 对于边界像素，直接保留梯度值
        nms_output[y * width + x] = current;
    }
}


// 双阈值处理核函数
__global__ void threshold_kernel(const float* nms, unsigned char* edges, int width, int height, float high_thresh, float low_thresh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float val = nms[y * width + x];
    if (val >= high_thresh) {
        edges[y * width + x] = 2; // 强边缘
    } else if (val >= low_thresh) {
        edges[y * width + x] = 1; // 弱边缘
    } else {
        edges[y * width + x] = 0; // 非边缘
    }
}

// 滞后阈值处理核函数
// __global__ void hysteresis_kernel(unsigned char* edges, int width, int height)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int total = width * height;
//     if (idx >= total) return;

//     if (edges[idx] != 2) return; // 只处理强边缘

//     int x = idx % width;
//     int y = idx / width;

//     // 创建一个栈用于跟踪弱边缘
//     int stack[1024];
//     int top = -1;

//     stack[++top] = idx;

//     while (top >= 0) {
//         int current_idx = stack[top--];
//         int cx = current_idx % width;
//         int cy = current_idx / width;

//         for (int ky = -1; ky <= 1; ++ky) {
//             for (int kx = -1; kx <=1; ++kx) {
//                 int nx = cx + kx;
//                 int ny = cy + ky;
//                 if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
//                     int neighbor_idx = ny * width + nx;
//                     if (edges[neighbor_idx] == 1) { // 弱边缘
//                         edges[neighbor_idx] = 2; // 升级为强边缘
//                         if (top < 1023) {
//                             stack[++top] = neighbor_idx;
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

__global__ void hysteresis_kernel(unsigned char* edges, const float* gradient, int width, int height, float low_thresh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;

    int idx = y * width + x;

    if (edges[idx] == 2) // 强边缘
    {
        // 检查八邻域
        for (int ky = -1; ky <= 1; ++ky)
        {
            for (int kx = -1; kx <=1; ++kx)
            {
                if (kx == 0 && ky == 0) continue;
                int nx = x + kx;
                int ny = y + ky;
                int n_idx = ny * width + nx;
                if (edges[n_idx] == 1 && gradient[n_idx] >= low_thresh)
                {
                    edges[n_idx] = 2; // 升级为强边缘
                }
            }
        }
    }
}


// 主函数
void canny_cuda(float* h_image, int width, int height, float sigma, float* h_output)
{
    float *d_image, *d_blurred, *d_gradient, *d_direction, *d_nms;
    unsigned char* d_edges;
    float* d_kernel;
    float* h_kernel;
    int kernel_radius;
    int image_size_f = width * height * sizeof(float);
    int image_size_uc = width * height * sizeof(unsigned char);

    // 生成高斯核
    generate_gaussian_kernel(sigma, &h_kernel, &kernel_radius);
    int kernel_size = (2 * kernel_radius + 1) * (2 * kernel_radius + 1) * sizeof(float);

    // 分配设备内存
    CUDA_CHECK(cudaMalloc((void**)&d_image, image_size_f));
    CUDA_CHECK(cudaMalloc((void**)&d_blurred, image_size_f));
    CUDA_CHECK(cudaMalloc((void**)&d_gradient, image_size_f));
    CUDA_CHECK(cudaMalloc((void**)&d_direction, image_size_f));
    CUDA_CHECK(cudaMalloc((void**)&d_nms, image_size_f));
    CUDA_CHECK(cudaMalloc((void**)&d_edges, image_size_uc));
    CUDA_CHECK(cudaMalloc((void**)&d_kernel, kernel_size));

    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_image, h_image, image_size_f, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice));

    // 定义块和网格大小
    dim3 block(16, 16);
    dim3 grid((width + block.x -1)/block.x, (height + block.y -1)/block.y);

    // 高斯滤波（共享内存大小）
    int shared_mem_size = (block.x + 2 * kernel_radius) * (block.y + 2 * kernel_radius) * sizeof(float);
    gaussian_filter_kernel<<<grid, block, shared_mem_size>>>(d_image, d_blurred, width, height, d_kernel, kernel_radius);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计算梯度幅值和方向
    sobel_filter_kernel<<<grid, block>>>(d_blurred, d_gradient, d_direction, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计算梯度幅值的最小值和最大值
    int total_pixels = width * height;
    int threads_per_block = 256;
    int blocks = (total_pixels + threads_per_block -1) / threads_per_block;

    float* d_min_vals;
    float* d_max_vals;
    CUDA_CHECK(cudaMalloc((void**)&d_min_vals, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_max_vals, blocks * sizeof(float)));

    min_max_kernel<<<blocks, threads_per_block>>>(d_gradient, d_min_vals, d_max_vals, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 复制最小值和最大值到主机
    float* h_min_vals = (float*)malloc(blocks * sizeof(float));
    float* h_max_vals = (float*)malloc(blocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_min_vals, d_min_vals, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max_vals, d_max_vals, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    // 在主机上计算全局最小值和最大值
    float h_min_gradient = FLT_MAX;
    float h_max_gradient = -FLT_MAX;
    for (int i = 0; i < blocks; ++i) {
        if (h_min_vals[i] < h_min_gradient) {
            h_min_gradient = h_min_vals[i];
        }
        if (h_max_vals[i] > h_max_gradient) {
            h_max_gradient = h_max_vals[i];
        }
    }

    free(h_min_vals);
    free(h_max_vals);
    CUDA_CHECK(cudaFree(d_min_vals));
    CUDA_CHECK(cudaFree(d_max_vals));

    // 计算梯度幅值的直方图
    int hist_size = 128;
    int* d_hist;
    CUDA_CHECK(cudaMalloc((void**)&d_hist, hist_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist, 0, hist_size * sizeof(int)));

    histogram_kernel<<<blocks, threads_per_block>>>(d_gradient, d_hist, width, height, h_min_gradient, h_max_gradient, hist_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 将直方图从设备端复制到主机端
    int* h_hist = (int*)malloc(hist_size * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, hist_size * sizeof(int), cudaMemcpyDeviceToHost));

    // 计算累积分布函数（CDF）
    int* h_cdf = (int*)malloc(hist_size * sizeof(int));
    h_cdf[0] = h_hist[0];
    for (int i = 1; i < hist_size; ++i) {
        h_cdf[i] = h_cdf[i - 1] + h_hist[i];
    }

    // 计算高阈值和低阈值对应的梯度幅值
    float high_thresh_ratio = 0.80f; // 90% 百分位
    float low_thresh_ratio = 0.60f;  // 10% 百分位
    int high_count = (int)(total_pixels * high_thresh_ratio);
    int low_count = (int)(total_pixels * low_thresh_ratio);

    float bin_size = (h_max_gradient - h_min_gradient + 1e-6f) / hist_size;

    float high_threshold = 0.0f;
    float low_threshold = 0.0f;

    for (int i = 0; i < hist_size; ++i) {
        if (h_cdf[i] >= low_count && low_threshold == 0.0f) {
            low_threshold = h_min_gradient + bin_size * i;
        }
        if (h_cdf[i] >= high_count && high_threshold == 0.0f) {
            high_threshold = h_min_gradient + bin_size * i;
            break;
        }
    }

    printf("Min Gradient: %f\n", h_min_gradient);
    printf("Max Gradient: %f\n", h_max_gradient);
    printf("Low Threshold: %f\n", low_threshold);
    printf("High Threshold: %f\n", high_threshold);

    free(h_hist);
    free(h_cdf);
    CUDA_CHECK(cudaFree(d_hist));

    float* h_gradient = (float*)malloc(image_size_f);
    CUDA_CHECK(cudaMemcpy(h_gradient, d_gradient, image_size_f, cudaMemcpyDeviceToHost));

    // 打印前十个梯度幅值
    for (int i = 0; i < 10; ++i) {
        printf("Gradient[%d] = %f\n", i, h_gradient[i]);
    }

    free(h_gradient);


    // 非极大值抑制
    non_max_suppression_kernel<<<grid, block>>>(d_gradient, d_direction, d_nms, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 从设备复制 NMS 输出到主机
    float* h_nms = (float*)malloc(image_size_f);
    CUDA_CHECK(cudaMemcpy(h_nms, d_nms, image_size_f, cudaMemcpyDeviceToHost));

    // 打印前十个 NMS 值
    for (int i = 0; i < 10; ++i) {
        printf("NMS[%d] = %f\n", i, h_nms[i]);
    }

    free(h_nms);

    // 双阈值处理
    threshold_kernel<<<grid, block>>>(d_nms, d_edges, width, height, high_threshold, low_threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 滞后阈值处理
    // hysteresis_kernel<<<blocks, threads_per_block>>>(d_edges, width, height);
    hysteresis_kernel<<<grid, block>>>(d_edges, d_gradient, width, height, low_threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 将结果复制回主机
    unsigned char* h_edges = (unsigned char*)malloc(image_size_uc);
    CUDA_CHECK(cudaMemcpy(h_edges, d_edges, image_size_uc, cudaMemcpyDeviceToHost));
    for (int i = 0; i < width * height; ++i) {
        h_output[i] = (h_edges[i] == 2) ? 1.0f : 0.0f;
    }
    free(h_edges);

    // 释放内存
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_gradient));
    CUDA_CHECK(cudaFree(d_direction));
    CUDA_CHECK(cudaFree(d_nms));
    CUDA_CHECK(cudaFree(d_edges));
    CUDA_CHECK(cudaFree(d_kernel));
    free(h_kernel);
}

}

