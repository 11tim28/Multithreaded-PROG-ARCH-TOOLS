#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <cmath>
#include <iomanip> // For std::setw
#include <pthread.h> // NEW: Using POSIX threads instead of std::thread

#include "ga_types.h"
#include "gpu_interface.h"

using namespace std;

#define NUM_GEN 5

// Define an enum to clearly specify the selection strategy
enum class SelectionStrategy {
    SORTED_TRUNCATION,
    RANDOM
};

// --- Structure to pass multiple arguments to the pthread function ---
// pthread_create only accepts a single void* argument, so we package the necessary data here.
struct ThreadArgs {
    SelectionStrategy strategy;
    // We pass a pointer to the vector so the thread can modify the shared result vector.
    std::vector<float>* best_scores_out; 
    int run_seed;
};

// --- Comparison function for sorting based on cost (fitness) ---
bool compareIndividuals(const Individual& a, const Individual& b) {
    return a.cost > b.cost;
}

Individual create_random_individual(default_random_engine& generator) {
    Individual ind;
    uniform_int_distribution<int> distribution(0, 1);
    
    for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
        ind.chromosome[i] = distribution(generator);
    }
    ind.cost = -1.0f; 
    return ind;
}

/**
 * @function crossover
 * @brief CPU Task: Performs Static Single-Point Crossover (at the midpoint).
 */
vector<Individual> crossover(const Individual& parent1, const Individual& parent2) {
    vector<Individual> children(2);
    
    // Initialize children by copying parents first
    children[0] = parent1;
    children[1] = parent2;

    // Determine the STATIC crossover point: The midpoint.
    const int crossover_point = CHROMOSOME_LENGTH / 2;
    
    // Perform Static Single-Point Crossover
    for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
        if (i >= crossover_point) {
            // After the crossover point, the genes are swapped:
            children[0].chromosome[i] = parent2.chromosome[i]; 
            children[1].chromosome[i] = parent1.chromosome[i]; 
        } 
    }
    
    // Reset costs since they are new, unevaluated individuals
    children[0].cost = -1.0f; 
    children[1].cost = -1.0f;
    
    return children; 
}

// CPU Task: Mutation - Placeholder
// void mutate(Individual& individual) {
//     // Placeholder implementation: no actual mutation
// }

void mutate(Individual& ind, default_random_engine& gen, float mutation_rate = 0.01f) {
    uniform_real_distribution<float> prob(0.0f, 1.0f);
    for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
        if (prob(gen) < mutation_rate)
            ind.chromosome[i] = 1 - ind.chromosome[i];
    }
}

/**
 * @function random_selection
 * @brief CPU Task: Selects one parent using pure random selection.
 */
int random_selection(const vector<Individual>& population, default_random_engine& generator) {
    uniform_int_distribution<int> distribution(0, population.size() - 1);
    return distribution(generator);
}


// --- Core GA Loop (Worker Function) ---
void run_ga(SelectionStrategy strategy, vector<float>& best_scores_out, int run_seed) {
    // Initialize the thread-local random number generator using the provided seed
    default_random_engine generator(run_seed); 

    // 1. Initialize Population
    vector<Individual> population;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population.push_back(create_random_individual(generator));
    }

    // --- CRITICAL PRE-LOOP STEP: Initial Cost Evaluation ---
    vector<float> initial_costs = evaluate_children_gpu(population);
    for (size_t i = 0; i < population.size(); ++i) {
        population[i].cost = initial_costs[i];
    }
    
    // Sort the population initially to find the best score
    sort(population.begin(), population.end(), compareIndividuals);
    best_scores_out.push_back(population.front().cost); // Store Gen 0 score


    for (int gen = 0; gen < NUM_GEN; ++gen) {
        // --- CPU PHASE 1: Selection, Crossover, Mutation ---
        vector<Individual> children_pool;
        
        
        if (strategy == SelectionStrategy::SORTED_TRUNCATION) {
            // Strategy 1: Sorted Truncation Selection
            sort(population.begin(), population.end(), compareIndividuals);
            
            for (int i = 0; i < POPULATION_SIZE / 2; ++i) {
                const Individual& parent1 = population[2 * i];
                const Individual& parent2 = population[2 * i + 1];
                
                vector<Individual> children = crossover(parent1, parent2);
                mutate(children[0], generator);
                mutate(children[1], generator);

                children_pool.push_back(children[0]);
                children_pool.push_back(children[1]);
            }
        } else {
            // Strategy 2: Pure Random Selection
             for (int i = 0; i < POPULATION_SIZE / 2; ++i) {
                int idx1 = random_selection(population, generator);
                int idx2 = random_selection(population, generator);
                
                // Ensure they are different individuals
                while (idx1 == idx2) {
                     idx2 = random_selection(population, generator);
                }

                const Individual& parent1 = population[idx1];
                const Individual& parent2 = population[idx2];
                
                vector<Individual> children = crossover(parent1, parent2);
                mutate(children[0], generator);
                mutate(children[1], generator);

                children_pool.push_back(children[0]);
                children_pool.push_back(children[1]);
            }
        }
        
        // --- CPU/GPU INTERFACE: Cost Evaluation Call (GPU SIMULATED) ---
        vector<float> costs = evaluate_children_gpu(children_pool); 


        // --- CPU PHASE 2: Reintegration & New Generation ---
        
        // 1. Update costs (Reintegration)
        for (size_t i = 0; i < children_pool.size(); ++i) {
            children_pool[i].cost = costs[i];
        }

        // 2. Survivor Selection (Generational replacement: children replace parents)
        population = children_pool; 

        // 3. Prepare for next generation (Sort to find best score)
        sort(population.begin(), population.end(), compareIndividuals);

        // Store the best cost for this generation
        best_scores_out.push_back(population.front().cost); 
    }
}

/**
 * @function run_ga_wrapper
 * @brief Required wrapper function for pthread_create.
 * @param args_ptr Pointer to the ThreadArgs structure.
 */
void* run_ga_wrapper(void* args_ptr) {
    ThreadArgs* args = static_cast<ThreadArgs*>(args_ptr);
    run_ga(args->strategy, *args->best_scores_out, args->run_seed);
    delete args; // Clean up arguments allocated in main
    return NULL;
}


// --- Placeholder Implementation for GPU Interface (This will be replaced by CUDA code) ---
// std::vector<float> evaluate_children_gpu(const std::vector<Individual>& children) {
//      std::vector<float> costs(children.size());
     
//      // The core logic (counting 1s) is simulated here:
//      for(size_t i = 0; i < children.size(); ++i) {
//          float sum = 0.0f;
//          for(int j = 0; j < CHROMOSOME_LENGTH; ++j) {
//              // Fitness is the sum of '1's, maximizing the cost.
//              sum += children[i].chromosome[j];
//          }
//          costs[i] = sum; 
//      }
//      return costs;
// }


int main() {
    // Containers to hold results from concurrent runs
    vector<float> sorted_results;
    vector<float> random_results;

    cout << "Starting concurrent GA runs (" << NUM_GEN << " generations each) on CPU threads using pthreads...\n";

    // Pthread declarations
    pthread_t sorted_thread_id;
    pthread_t random_thread_id;
    
    // 1. Prepare arguments for the Sorted Truncation thread
    ThreadArgs* sorted_args = new ThreadArgs{
        SelectionStrategy::SORTED_TRUNCATION,
        &sorted_results,
        (int)time(0)
    };

    // 2. Prepare arguments for the Pure Random thread
    ThreadArgs* random_args = new ThreadArgs{
        SelectionStrategy::RANDOM,
        &random_results,
        (int)time(0) + 1 // Use a different seed
    };
    
    // 3. Launch the threads
    pthread_create(&sorted_thread_id, NULL, run_ga_wrapper, sorted_args);
    pthread_create(&random_thread_id, NULL, run_ga_wrapper, random_args);

    // 4. Wait for both threads to complete (Join)
    pthread_join(sorted_thread_id, NULL);
    pthread_join(random_thread_id, NULL);

    // --- Final Comparison Report ---
    cout << "\n========================================================================\n";
    cout << "                         CONCURRENT GA COMPARISON                       \n";
    cout << "========================================================================\n";
    cout << "Generation | Sorted Truncation Best Cost | Pure Random Best Cost\n";
    cout << "-----------|-----------------------------|---------------------\n";
    
    size_t num_generations = min(sorted_results.size(), random_results.size());

    for (size_t i = 0; i < num_generations; ++i) {
        if(i > 0 && i < num_generations-1) continue;
        cout << setw(10) << i << " | "
             << setw(27) << sorted_results[i] << " | "
             << setw(19) << random_results[i] << endl;

        // Stop printing if the optimal score is reached in the Sorted run
        if (sorted_results[i] >= CHROMOSOME_LENGTH) {
             if (i < num_generations - 1) {
                 cout << "--- OPTIMUM REACHED (Sorted) ---\n";
             }
             break; 
        }
    }
    cout << "========================================================================\n";
    cout << "Final Best Score (Sorted Truncation): " << sorted_results.back() << " / " << CHROMOSOME_LENGTH << endl;
    cout << "Final Best Score (Pure Random):       " << random_results.back() << " / " << CHROMOSOME_LENGTH << endl;
    
    return 0;
}
