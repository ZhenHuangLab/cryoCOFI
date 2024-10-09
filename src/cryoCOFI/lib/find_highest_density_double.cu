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
        atomicAdd(d_sum, s_sum[0]);
        atomicAdd(d_sum_sq, s_sum_sq[0]);
    }
}

__global__ void normalizeAndComputeHistogram(double* d_img, int* d_mask, int* d_inside_hist, int* d_outside_hist, int size, double mean, double std_dev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double val = d_img[idx];
        double normalized_val;
        
        if (std_dev != 0.0) {
            normalized_val = (val - mean) / std_dev;
        } else {
            normalized_val = val - mean; // If std_dev is 0, just subtract mean
        }
        
        // Scale to [0, 1] range
        normalized_val = (normalized_val + 3) / 6; // Assuming most values fall within 3 standard deviations
        normalized_val = fmin(1.0, fmax(0.0, normalized_val)); // Clamp to [0, 1]
        
        int bin = (int)(normalized_val * 255 + 0.5); // Add 0.5 for proper rounding
        bin = min(255, max(0, bin));
        
        if (d_mask[idx] == 1) {
            atomicAdd(&d_inside_hist[bin], 1);
        } else if (d_mask[idx] == 0) {
            atomicAdd(&d_outside_hist[bin], 1);
        }
    }
}

extern "C" {
    double find_highest_density_cuda(double* h_img, int* h_mask, int size) {
        double *d_img;
        int *d_mask, *d_inside_hist, *d_outside_hist;
        
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
        normalizeAndComputeHistogram<<<numBlocks, blockSize>>>(d_img, d_mask, d_inside_hist, d_outside_hist, size, mean, std_dev);
        
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
        
        return diff;
    }
}