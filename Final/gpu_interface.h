#ifndef GPU_INTERFACE_H
#define GPU_INTERFACE_H

#include "ga_types.h"
#include <vector>

/**
 * @function evaluate_children_gpu
 * @brief CPU function that interfaces with the CUDA runtime to evaluate the cost of children in parallel.
 * * This is the critical separation point. When the CPU calls this, it expects:
 * 1. Data transfer: Host (CPU) -> Device (GPU).
 * 2. Parallel execution: The CUDA kernel runs (GPU).
 * 3. Results transfer: Device (GPU) -> Host (CPU).
 * * @param children A vector containing the new individuals (children) whose costs need evaluation.
 * @return A vector of floats representing the calculated costs.
 */
std::vector<float> evaluate_children_gpu(const std::vector<Individual>& children);

#endif // GPU_INTERFACE_H
