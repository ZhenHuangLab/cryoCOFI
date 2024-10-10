#include <cuda_runtime.h>

__device__ double atomicAdd_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void computeMeanAndStdDev(double* d_img, int size, double* d_sum, double* d_sum_sq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ double shared_mem[];
    double* s_sum = shared_mem;
    double* s_sum_sq = &s_sum[blockDim.x];

    s_sum[threadIdx.x] = 0.0;
    s_sum_sq[threadIdx.x] = 0.0;

    if (idx < size) {
        double val = d_img[idx];
        s_sum[threadIdx.x] = val;
        s_sum_sq[threadIdx.x] = val * val;
    }

    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd_double(d_sum, s_sum[0]);
        atomicAdd_double(d_sum_sq, s_sum_sq[0]);
    }
}

__global__ void normalizeAndComputeHistogram(double* d_img, int* d_mask, int* d_inside_hist, int* d_outside_hist, int size, double mean, double std_dev, int* d_inside_flags, int n_inside_flags, int* d_outside_flags, int n_outside_flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double val = d_img[idx];
        double normalized_val;
        
        if (std_dev != 0.0) {
            normalized_val = (val - mean) / std_dev;
        } else {
            normalized_val = val - mean;
        }
        
        normalized_val = (normalized_val + 3) / 6;
        normalized_val = fmin(1.0, fmax(0.0, normalized_val));
        
        int bin = (int)(normalized_val * 255 + 0.5);
        bin = min(255, max(0, bin));
        
        int mask_value = d_mask[idx];
        bool is_inside = false;
        bool is_outside = false;

        for (int i = 0; i < n_inside_flags; i++) {
            if (mask_value == d_inside_flags[i]) {
                is_inside = true;
                break;
            }
        }

        for (int i = 0; i < n_outside_flags; i++) {
            if (mask_value == d_outside_flags[i]) {
                is_outside = true;
                break;
            }
        }

        if (is_inside) {
            atomicAdd(&d_inside_hist[bin], 1);
        }
        if (is_outside) {
            atomicAdd(&d_outside_hist[bin], 1);
        }
    }
}

extern "C" {
    double find_highest_density_cuda(double* h_img, int* h_mask, int size, int* h_inside_flags, int n_inside_flags, int* h_outside_flags, int n_outside_flags) {
        double *d_img;
        int *d_mask, *d_inside_hist, *d_outside_hist;

        int *d_inside_flags, *d_outside_flags;
        cudaMalloc(&d_inside_flags, n_inside_flags * sizeof(int));
        cudaMalloc(&d_outside_flags, n_outside_flags * sizeof(int));
        cudaMemcpy(d_inside_flags, h_inside_flags, n_inside_flags * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_outside_flags, h_outside_flags, n_outside_flags * sizeof(int), cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_img, size * sizeof(double));
        cudaMalloc(&d_mask, size * sizeof(int));
        cudaMalloc(&d_inside_hist, 256 * sizeof(int));
        cudaMalloc(&d_outside_hist, 256 * sizeof(int));
        
        cudaMemcpy(d_img, h_img, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mask, h_mask, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_inside_hist, 0, 256 * sizeof(int));
        cudaMemset(d_outside_hist, 0, 256 * sizeof(int));
        
        // Compute mean and standard deviation
        double *d_sum, *d_sum_sq;
        cudaMalloc(&d_sum, sizeof(double));
        cudaMalloc(&d_sum_sq, sizeof(double));
        cudaMemset(d_sum, 0, sizeof(double));
        cudaMemset(d_sum_sq, 0, sizeof(double));

        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        int sharedMemSize = 2 * blockSize * sizeof(double);

        computeMeanAndStdDev<<<numBlocks, blockSize, sharedMemSize>>>(d_img, size, d_sum, d_sum_sq);

        double h_sum, h_sum_sq;
        cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(double), cudaMemcpyDeviceToHost);

        double mean = h_sum / size;
        double variance = (h_sum_sq / size) - (mean * mean);
        double std_dev = sqrt(variance);

        // Normalize and compute histogram
        normalizeAndComputeHistogram<<<numBlocks, blockSize>>>(d_img, d_mask, d_inside_hist, d_outside_hist, size, mean, std_dev, d_inside_flags, n_inside_flags, d_outside_flags, n_outside_flags);
        
        int h_inside_hist[256], h_outside_hist[256];
        cudaMemcpy(h_inside_hist, d_inside_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_outside_hist, d_outside_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);
        
        int inside_max_idx = 0, outside_max_idx = 0;
        for (int i = 1; i < 256; i++) {
            if (h_inside_hist[i] > h_inside_hist[inside_max_idx]) inside_max_idx = i;
            if (h_outside_hist[i] > h_outside_hist[outside_max_idx]) outside_max_idx = i;
        }
        
        double inside_density = (double)inside_max_idx / 255.0;
        double outside_density = (double)outside_max_idx / 255.0;
        double diff = inside_density - outside_density;
        
        cudaFree(d_sum);
        cudaFree(d_sum_sq);
        
        cudaFree(d_img);
        cudaFree(d_mask);
        cudaFree(d_inside_hist);
        cudaFree(d_outside_hist);

        cudaFree(d_inside_flags);
        cudaFree(d_outside_flags);
        
        return diff;
    }
}