// hough_transform.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>

extern "C" {

// Define constants
#define NUM_THETA 360

__constant__ float d_cos_theta[NUM_THETA];
__constant__ float d_sin_theta[NUM_THETA];

// CUDA kernel: Compute the Hough accumulator with weights
__global__ void houghTransformKernel(
    const unsigned char* edge_image,
    int rows, int cols,
    int r,
    int hough_max_x, int hough_max_y,
    int* accumulator)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = rows * cols;

    if (idx >= total_pixels)
        return;

    int y = idx / cols;
    int x = idx % cols;

    if (edge_image[y * cols + x] > 0)
    {
        // Calculate the number of edge pixels in the neighborhood
        int edge_count = 0;
        for (int dy = -3; dy <= 3; ++dy)
        {
            for (int dx = -3; dx <= 3; ++dx)
            {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < cols && ny >= 0 && ny < rows)
                {
                    if (edge_image[ny * cols + nx] > 0)
                    {
                        edge_count++;
                    }
                }
            }
        }

        // Calculate the weight; more edge pixels in the neighborhood lead to a higher weight
        float weight = edge_count / 9.0f; // Maximum value is 1

        // For each theta value
        for (int theta_idx = 0; theta_idx < NUM_THETA; ++theta_idx)
        {
            float cos_t = d_cos_theta[theta_idx];
            float sin_t = d_sin_theta[theta_idx];

            float a = x - r * cos_t;
            float b = y - r * sin_t;

            int a_idx = (int)roundf(a + r);
            int b_idx = (int)roundf(b + r);

            if (a_idx >= 0 && a_idx < hough_max_x && b_idx >= 0 && b_idx < hough_max_y)
            {
                int vote = (int)(weight * 100); // Convert weight to integer votes
                atomicAdd(&accumulator[b_idx * hough_max_x + a_idx], vote);
            }
        }
    }
}

// CUDA kernel: Find the maximum value in the accumulator
__global__ void findAccumulatorMaxKernel(
    const int* accumulator,
    int accumulator_size,
    int* max_value)
{
    extern __shared__ int shared_max[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Initialize shared memory
    shared_max[tid] = 0;

    // Load data into shared memory
    if (idx < accumulator_size)
    {
        shared_max[tid] = accumulator[idx];
    }

    __syncthreads();

    // Reduction to find the maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride && (idx + stride) < accumulator_size)
        {
            if (shared_max[tid] < shared_max[tid + stride])
            {
                shared_max[tid] = shared_max[tid + stride];
            }
        }
        __syncthreads();
    }

    // Write the maximum value of each block to global memory
    if (tid == 0)
    {
        atomicMax(max_value, shared_max[0]);
    }
}

// CUDA kernel: Collect positions with the maximum value
__global__ void collectMaxPositionsKernel(
    const int* accumulator,
    int hough_max_x, int hough_max_y,
    int max_value,
    int* max_positions, // Array to store positions [a1, b1, a2, b2, ...]
    int* max_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = hough_max_x * hough_max_y;

    if (idx >= total_size)
        return;

    int value = accumulator[idx];

    if (value == max_value)
    {
        int pos = atomicAdd(max_count, 1);
        max_positions[2 * pos] = idx % hough_max_x;
        max_positions[2 * pos + 1] = idx / hough_max_x;
    }
}

// CUDA kernel: Compute the average in a 7x7 neighborhood
__global__ void computeNeighborhoodAveragesKernel(
    const int* accumulator,
    int hough_max_x, int hough_max_y,
    int* max_positions,
    int max_count,
    float* averages)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= max_count)
        return;

    int a = max_positions[2 * idx];
    int b = max_positions[2 * idx + 1];

    int sum = 0;
    int count = 0;

    for (int dy = -3; dy <= 3; ++dy)
    {
        for (int dx = -3; dx <= 3; ++dx)
        {
            int x = a + dx;
            int y = b + dy;
            if (x >= 0 && x < hough_max_x && y >= 0 && y < hough_max_y)
            {
                sum += accumulator[y * hough_max_x + x];
                count++;
            }
        }
    }

    averages[idx] = (float)sum / count;
}

// Host function for Python to call
__host__ void hough_transform_for_radius(
    const unsigned char* edge_image,
    int rows, int cols,
    int r,
    int* best_a, int* best_b,
    int* accumulator_out)
{
    int hough_max_x = 2 * r + cols;
    int hough_max_y = 2 * r + rows;

    int accumulator_size = hough_max_x * hough_max_y;

    // Allocate device memory
    unsigned char* d_edge_image;
    int* d_accumulator;

    // Macro to check CUDA errors
    #define CHECK_CUDA_ERROR(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
                return; \
            } \
        } while (0)

    // Allocate d_edge_image
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_edge_image, rows * cols * sizeof(unsigned char)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_edge_image, edge_image, rows * cols * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Allocate d_accumulator
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_accumulator, accumulator_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_accumulator, 0, accumulator_size * sizeof(int)));

    // Initialize cos and sin arrays
    float h_theta[NUM_THETA];
    for (int i = 0; i < NUM_THETA; ++i)
    {
        float theta = i * 2.0f * M_PI / NUM_THETA;
        h_theta[i] = theta;
    }

    float h_cos_theta[NUM_THETA];
    float h_sin_theta[NUM_THETA];

    for (int i = 0; i < NUM_THETA; ++i)
    {
        h_cos_theta[i] = cosf(h_theta[i]);
        h_sin_theta[i] = sinf(h_theta[i]);
    }

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_cos_theta, h_cos_theta, NUM_THETA * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_sin_theta, h_sin_theta, NUM_THETA * sizeof(float)));

    // Launch the Hough transform kernel
    int total_pixels = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    houghTransformKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_edge_image, rows, cols, r, hough_max_x, hough_max_y, d_accumulator);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Find the maximum value in the accumulator
    int* d_max_value;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_max_value, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_max_value, 0, sizeof(int)));

    int accumulator_blocks = (accumulator_size + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(int);

    findAccumulatorMaxKernel<<<accumulator_blocks, threadsPerBlock, sharedMemSize>>>(
        d_accumulator, accumulator_size, d_max_value);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    int h_max_value = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_max_value, d_max_value, sizeof(int), cudaMemcpyDeviceToHost));

    // Collect positions with the maximum value
    int* d_max_positions;
    int* d_max_count;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_max_positions, accumulator_size * 2 * sizeof(int))); // Worst case: all positions are max
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_max_count, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_max_count, 0, sizeof(int)));

    collectMaxPositionsKernel<<<accumulator_blocks, threadsPerBlock>>>(
        d_accumulator, hough_max_x, hough_max_y, h_max_value, d_max_positions, d_max_count);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    int h_max_count = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_max_count, d_max_count, sizeof(int), cudaMemcpyDeviceToHost));

    // printf("Number of positions with max value: %d\n", h_max_count);

    if (h_max_count == 0)
    {
        fprintf(stderr, "No positions with max value found.\n");
        *best_a = -1;
        *best_b = -1;
    }
    else if (h_max_count == 1)
    {
        int h_max_positions[2];
        CHECK_CUDA_ERROR(cudaMemcpy(h_max_positions, d_max_positions, 2 * sizeof(int), cudaMemcpyDeviceToHost));
        *best_a = h_max_positions[0];
        *best_b = h_max_positions[1];
    }
    else
    {
        // Compute neighborhood averages for each maximum position
        float* d_averages;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_averages, h_max_count * sizeof(float)));

        int avg_threadsPerBlock = 256;
        int avg_blocksPerGrid = (h_max_count + avg_threadsPerBlock - 1) / avg_threadsPerBlock;

        computeNeighborhoodAveragesKernel<<<avg_blocksPerGrid, avg_threadsPerBlock>>>(
            d_accumulator, hough_max_x, hough_max_y, d_max_positions, h_max_count, d_averages);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Copy averages back to host
        float* h_averages = new float[h_max_count];
        CHECK_CUDA_ERROR(cudaMemcpy(h_averages, d_averages, h_max_count * sizeof(float), cudaMemcpyDeviceToHost));

        // Copy positions back to host
        int* h_max_positions = new int[h_max_count * 2];
        CHECK_CUDA_ERROR(cudaMemcpy(h_max_positions, d_max_positions, h_max_count * 2 * sizeof(int), cudaMemcpyDeviceToHost));

        // Find the position with the highest average
        int best_idx = 0;
        float max_average = h_averages[0];
        for (int i = 1; i < h_max_count; ++i)
        {
            if (h_averages[i] > max_average)
            {
                max_average = h_averages[i];
                best_idx = i;
            }
        }

        *best_a = h_max_positions[2 * best_idx];
        *best_b = h_max_positions[2 * best_idx + 1];

        // Free host memory
        delete[] h_averages;
        delete[] h_max_positions;
        CHECK_CUDA_ERROR(cudaFree(d_averages));
    }

    // Copy the accumulator back to host if requested
    if (accumulator_out != NULL)
    {
        CHECK_CUDA_ERROR(cudaMemcpy(accumulator_out, d_accumulator, accumulator_size * sizeof(int), cudaMemcpyDeviceToHost));
    }

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_edge_image));
    CHECK_CUDA_ERROR(cudaFree(d_accumulator));
    CHECK_CUDA_ERROR(cudaFree(d_max_value));
    CHECK_CUDA_ERROR(cudaFree(d_max_positions));
    CHECK_CUDA_ERROR(cudaFree(d_max_count));
}

}
