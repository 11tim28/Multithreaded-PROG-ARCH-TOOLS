// gpu_cost.cu
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "gpu_interface.h"
#include "ga_types.h"  // must define Individual and CHROMOSOME_LENGTH

// Simple error-check macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        /* abort or throw is fine for debugging */ \
        exit(EXIT_FAILURE); \
    } \
} while(0)


// Kernel: each thread sums one individual's chromosome of length `chrom_len`
// chrom is layout: [ind0_gene0, ind0_gene1, ..., ind0_geneL-1, ind1_gene0, ...]
__global__
void compute_cost_kernel(const int* chrom, float* costs, int chrom_len, int num_individuals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_individuals) return;

    int base = idx * chrom_len;
    int sum = 0;
    // simple loop; chrom_len expected to be small (e.g., 100s)
    for (int k = 0; k < chrom_len; ++k) {
        sum += chrom[base + k];  // genes are 0 or 1
    }
    costs[idx] = static_cast<float>(sum);
}


// Host wrapper called by your C++ program
std::vector<float> evaluate_children_gpu(const std::vector<Individual>& children) {
    const int num = static_cast<int>(children.size());
    const int chrom_len = CHROMOSOME_LENGTH;

    std::vector<float> host_costs(num, 0.0f);
    if (num == 0) return host_costs;

    // 1) Flatten chromosomes into a contiguous host buffer of ints
    std::vector<int> host_chrom(num * chrom_len);
    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < chrom_len; ++j) {
            // Be defensive: assume chromosome entries are 0/1 ints
            host_chrom[i * chrom_len + j] = static_cast<int>(children[i].chromosome[j]);
        }
    }

    // 2) Allocate device memory
    int* dev_chrom = nullptr;
    float* dev_costs = nullptr;
    size_t chrom_bytes = sizeof(int) * host_chrom.size();
    size_t cost_bytes = sizeof(float) * num;

    CUDA_CHECK(cudaMalloc((void**)&dev_chrom, chrom_bytes));
    CUDA_CHECK(cudaMalloc((void**)&dev_costs, cost_bytes));

    // 3) Copy chromosomes to device
    CUDA_CHECK(cudaMemcpy(dev_chrom, host_chrom.data(), chrom_bytes, cudaMemcpyHostToDevice));

    // 4) Launch kernel
    const int threads_per_block = 256;
    const int blocks = (num + threads_per_block - 1) / threads_per_block;
    compute_cost_kernel<<<blocks, threads_per_block>>>(dev_chrom, dev_costs, chrom_len, num);

    // check for launch errors and device-side errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5) Copy costs back
    CUDA_CHECK(cudaMemcpy(host_costs.data(), dev_costs, cost_bytes, cudaMemcpyDeviceToHost));

    // 6) Free device memory
    CUDA_CHECK(cudaFree(dev_chrom));
    CUDA_CHECK(cudaFree(dev_costs));

    return host_costs;
}
