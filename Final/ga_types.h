#ifndef GA_TYPES_H
#define GA_TYPES_H

// --- System Configuration ---
// These are constant values used by both the CPU and GPU code.
const int CHROMOSOME_LENGTH = 10000;
const int POPULATION_SIZE = 1024;
const float MUTATION_RATE = 0.01f;
const int NUM_GEN = 10;

/**
 * @struct Individual
 * @brief Represents a single individual in the genetic algorithm population.
 * * This structure will be mapped to device memory in CUDA.
 * It must be simple (POD) to ensure efficient memory transfer.
 */
struct Individual {
    // Chromosome: A binary string (20 bits)
    // Using an array of integers for simplicity in both C++ and CUDA kernel.
    int chromosome[CHROMOSOME_LENGTH]; 
    
    // Cost/Fitness value: What the GPU is responsible for calculating.
    // float is standard for fitness scores.
    float cost; 
};

#endif // GA_TYPES_H
