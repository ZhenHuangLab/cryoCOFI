#include <cuda_runtime.h>
#include <stdio.h>

__global__ void normalizeAndComputeHistogram(float* d_img, int* d_mask, int* d_inside_hist, int* d_outside_hist, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = d_img[idx];
        float normalized_val;
        
        if (max_val > min_val) {
            normalized_val = (val - min_val) / (max_val - min_val);
        } else {
            normalized_val = val; // If max_val == min_val, keep original value
        }
        
        int bin = (int)(normalized_val * 255);
        bin = min(255, max(0, bin));
        
        if (d_mask[idx] == 1) {
            atomicAdd(&d_inside_hist[bin], 1);
        } else if (d_mask[idx] == 0) {
            atomicAdd(&d_outside_hist[bin], 1);
        }
    }
}

extern "C" {
    float find_highest_density_cuda(float* h_img, int* h_mask, int size) {
        float *d_img;
        int *d_mask, *d_inside_hist, *d_outside_hist;
        
        cudaMalloc(&d_img, size * sizeof(float));
        cudaMalloc(&d_mask, size * sizeof(int));
        cudaMalloc(&d_inside_hist, 256 * sizeof(int));
        cudaMalloc(&d_outside_hist, 256 * sizeof(int));
        
        cudaMemcpy(d_img, h_img, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mask, h_mask, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_inside_hist, 0, 256 * sizeof(int));
        cudaMemset(d_outside_hist, 0, 256 * sizeof(int));
        
        // Find min and max values
        float min_val = h_img[0], max_val = h_img[0];
        for (int i = 1; i < size; i++) {
            if (h_img[i] < min_val) min_val = h_img[i];
            if (h_img[i] > max_val) max_val = h_img[i];
        }
        
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        normalizeAndComputeHistogram<<<numBlocks, blockSize>>>(d_img, d_mask, d_inside_hist, d_outside_hist, size, min_val, max_val);
        
        int h_inside_hist[256], h_outside_hist[256];
        cudaMemcpy(h_inside_hist, d_inside_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_outside_hist, d_outside_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);
        
        int inside_max_idx = 0, outside_max_idx = 0;
        for (int i = 1; i < 256; i++) {
            if (h_inside_hist[i] > h_inside_hist[inside_max_idx]) inside_max_idx = i;
            if (h_outside_hist[i] > h_outside_hist[outside_max_idx]) outside_max_idx = i;
        }
        
        float inside_density = (float)inside_max_idx / 255.0;
        float outside_density = (float)outside_max_idx / 255.0;
        float diff = inside_density - outside_density;
        
        cudaFree(d_img);
        cudaFree(d_mask);
        cudaFree(d_inside_hist);
        cudaFree(d_outside_hist);
        
        return diff;
    }
}