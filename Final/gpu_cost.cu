#include <cuda_runtime.h>
#include "ga_types.h"
#include <vector>
#include <cstdio>

typedef int8_t gene_t;

__global__
void cost_kernel(const gene_t* d_chromosomes, float* d_costs, int num_children) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_children) return;

    float sum = 0;
    const gene_t* child = d_chromosomes + idx * CHROMOSOME_LENGTH;
    for (int j = 0; j < CHROMOSOME_LENGTH; ++j)
        sum += child[j];
    d_costs[idx] = sum;
}

std::vector<float> evaluate_children_gpu(const std::vector<Individual>& children, float* gpu_time)
{
    int N = children.size();
    std::vector<float> costs(N);

    // -----------------------------
    // Allocate host staging buffer
    // -----------------------------
    std::vector<gene_t> h_chromosomes(N * CHROMOSOME_LENGTH);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < CHROMOSOME_LENGTH; ++j)
            h_chromosomes[i * CHROMOSOME_LENGTH + j] = children[i].chromosome[j];

    gene_t *d_chromosomes;
    float *d_costs;
    cudaMalloc(&d_chromosomes, sizeof(gene_t) * N * CHROMOSOME_LENGTH);
    cudaMalloc(&d_costs, sizeof(float) * N);

    cudaMemcpy(d_chromosomes, h_chromosomes.data(),
               sizeof(gene_t) * N * CHROMOSOME_LENGTH, cudaMemcpyHostToDevice);

    // ==================================================
    //  ADD CUDA EVENT TIMING HERE
    // ==================================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);  // <<< START TIMING >>>

    // Launch kernel
    int block = 128;
    int grid = (N + block - 1) / block;
    cost_kernel<<<grid, block>>>(d_chromosomes, d_costs, N);

    cudaEventRecord(stop);   // <<< STOP TIMING >>>
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // printf("GPU kernel execution time = %.4f ms\n", milliseconds);
    *gpu_time = milliseconds;

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // -----------------------------
    // Copy back results
    // -----------------------------
    cudaMemcpy(costs.data(), d_costs, sizeof(float) * N, cudaMemcpyDeviceToHost);

    cudaFree(d_chromosomes);
    cudaFree(d_costs);

    return costs;
}
