#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <cmath>
#include <iomanip> // For std::setw
#include <pthread.h> // NEW: Using POSIX threads instead of std::thread
#include <time.h>
#include <stdio.h>
#include <valarray>     // std::valarray, std::log(valarray)
#include <fstream>

#include "ga_types.h"
#include "gpu_interface.h"

using namespace std;

// #define NUM_GEN 10

double get_time_ms(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Define an enum to clearly specify the selection strategy
enum class SelectionStrategy {
    SORTED_TRUNCATION,
    RANDOM
};

// --- Structure to pass multiple arguments to the pthread function ---
// pthread_create only accepts a single void* argument, so we package the necessary data here.
struct ThreadArgs {
    int tid;
    int num_threads;
    int run_seed;
    SelectionStrategy strategy;
    const vector<Individual>* population;
    vector<Individual>* children_pool;
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

inline void small_busy_wait(int iterations = 2000) {
    volatile float x = 0.0f;
    for (int i = 0; i < iterations; i++) {
        x = x * 1.01f + 0.0001f;  // meaningless math
    }
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
        small_busy_wait();
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

vector<Individual> selection(SelectionStrategy strategy, const vector<Individual>& population, int idx, int seed){
    vector<Individual> ans;
    default_random_engine generator(seed); 
    if(strategy == SelectionStrategy::SORTED_TRUNCATION){
        ans.push_back(population[2*idx]);
        ans.push_back(population[2*idx+1]);
        return ans;
    }
    else{
        int idx1 = random_selection(population, generator);
        int idx2 = random_selection(population, generator);

        while (idx1 == idx2) {
            idx2 = random_selection(population, generator);
        }
        ans.push_back(population[idx1]);
        ans.push_back(population[idx2]);
        return ans;
    }
}

void* ga_cpu_thread(void* arg){
    ThreadArgs* args = (ThreadArgs*)arg;
    int tid = args->tid;
    int num_threads = args->num_threads;
    int run_seed = args->run_seed;
    SelectionStrategy strategy = args->strategy;
    const vector<Individual>& population = *(args->population);
    vector<Individual>& children_pool = *(args->children_pool);

    default_random_engine generator(run_seed + tid * 99991);

    int half = POPULATION_SIZE / 2;

    int chunk = half / num_threads;
    int start = tid * chunk;
    int end = (tid == num_threads - 1) ? half : start + chunk;

    for(int i = start; i < end; i++){
        vector<Individual> parents = selection(strategy, population, i, run_seed + tid);
        const Individual& parent1 = parents[0];
        const Individual& parent2 = parents[1];
        vector<Individual> children = crossover(parent1, parent2);
        mutate(children[0], generator);
        mutate(children[1], generator);
        children_pool[2*i] = children[0];
        children_pool[2*i+1] = children[1];
    }
}

std::vector<float> evaluate_children_gpu(const std::vector<Individual>& children, float* gpu_time) {
    double start = get_time_ms();
     std::vector<float> costs(children.size());

     // The core logic (counting 1s) is simulated here:
     for(size_t i = 0; i < children.size(); ++i) {
         float sum = 0.0f;
         for(int j = 0; j < CHROMOSOME_LENGTH; ++j) {
             // Fitness is the sum of '1's, maximizing the cost.
             sum += children[i].chromosome[j];
         }
         costs[i] = sum; 
     }
     double end = get_time_ms();
     *gpu_time = end - start;
     return costs;
}

// heavy CPU cost function
static const int HEAVY_ITERS = 500;

std::vector<float> evaluate_children_gpu_heavy(const std::vector<Individual>& children, float* gpu_time) {
    double start = get_time_ms();
    std::vector<float> costs(children.size());

    for (size_t i = 0; i < children.size(); ++i) {
        float sum = 0.0f;

        for (int j = 0; j < CHROMOSOME_LENGTH; j++) {
            int bit = children[i].chromosome[j];

            // Skip if bit = 0 (still faster)
            if (bit == 0) continue;

            float acc = 0.0f;

            // Heavy math to stress CPU/GPU
            for (int k = 1; k <= HEAVY_ITERS; k++) {
                float a = std::sin(j * 0.01f * k);
                float b = std::cos((j + 1) * 0.02f * k);
                float c = std::log(1.0f + j * k * 0.0001f);

                acc += a * b + c;
            }

            sum += acc;
        }

        costs[i] = sum;
    }

    double end = get_time_ms();
    *gpu_time = end - start;
    return costs;
}



// --- Core GA Loop (Worker Function) ---
void run_ga_singlethread(SelectionStrategy strategy, vector<float>& best_scores_out, int run_seed, float* cpu_time, float* gpu_latency, Individual* global_best_individual, vector<float>& global_best_scores) {
    // Initialize the thread-local random number generator using the provided seed
    default_random_engine generator(run_seed); 

    // 1. Initialize Population
    vector<Individual> population;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population.push_back(create_random_individual(generator));
    }
    float gpu_time;

    // --- CRITICAL PRE-LOOP STEP: Initial Cost Evaluation ---
    // vector<float> initial_costs = evaluate_children_gpu(population, &gpu_time);
    vector<float> initial_costs = evaluate_children_gpu_heavy(population, &gpu_time);
    for (size_t i = 0; i < population.size(); ++i) {
        population[i].cost = initial_costs[i];
    }
    
    // Sort the population initially to find the best score
    sort(population.begin(), population.end(), compareIndividuals);
    best_scores_out.push_back(population.front().cost); // Store Gen 0 score    // Global-best tracking
    float global_best_cost = population.front().cost;
    // global_best_scores.push_back(global_best_cost);
    if (global_best_individual)
      *global_best_individual = population.front();
    
    vector<double> cpu_execution;
    vector<float> gpu_execution;



    for (int gen = 0; gen < NUM_GEN; ++gen) {
        double start = get_time_ms();
        // --- CPU PHASE 1: Selection, Crossover, Mutation ---
        vector<Individual> children_pool;
        
        vector<Individual> parents;
        
        if (strategy == SelectionStrategy::SORTED_TRUNCATION) {
            // Strategy 1: Sorted Truncation Selection
            sort(population.begin(), population.end(), compareIndividuals);
            
            
            for (int i = 0; i < POPULATION_SIZE / 2; ++i) {
                parents = selection(strategy, population, i, run_seed);
                const Individual& parent1 = parents[0];
                const Individual& parent2 = parents[1];
                
                vector<Individual> children = crossover(parent1, parent2);
                mutate(children[0], generator);
                mutate(children[1], generator);

                children_pool.push_back(children[0]);
                children_pool.push_back(children[1]);
                // children_pool[2*i] = children[0];
                // children_pool[2*i+1] = children[1];
            }
        } else {
            // Strategy 2: Pure Random Selection
             for (int i = 0; i < POPULATION_SIZE / 2; ++i) {
                parents = selection(strategy, population, i, run_seed);
                const Individual& parent1 = parents[0];
                const Individual& parent2 = parents[1];
                
                vector<Individual> children = crossover(parent1, parent2);
                mutate(children[0], generator);
                mutate(children[1], generator);

                children_pool.push_back(children[0]);
                children_pool.push_back(children[1]);
                // children_pool[2*i] = children[0];
                // children_pool[2*i+1] = children[1];
            }
        }
        double end = get_time_ms();
        // cout << "CPU time (single thread): " << end - start << " ms." << endl;
        cpu_execution.push_back(end-start);
        
        // --- CPU/GPU INTERFACE: Cost Evaluation Call (GPU SIMULATED) ---
        // vector<float> costs = evaluate_children_gpu(children_pool, &gpu_time); 
        vector<float> costs = evaluate_children_gpu_heavy(children_pool, &gpu_time); 
        gpu_execution.push_back(gpu_time);


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
        // --- Global best update (monotonic) ---
        if (population.front().cost > global_best_cost) {
            global_best_cost   = population.front().cost;
            if (global_best_individual)
              *global_best_individual = population.front();
        }

        // Store global best cost for convergence curve
        global_best_scores.push_back(global_best_cost); 
    }
    float tmp = 0;
    for(auto& a : cpu_execution){
        tmp += a;
    }
    tmp /= cpu_execution.size();
    *cpu_time = tmp;
    tmp = 0;
    for(auto& b : gpu_execution){
        tmp += b;
    }
    tmp /= gpu_execution.size();
    *gpu_latency = tmp;
}

// --- Core GA Loop (Worker Function) ---
void run_ga_multithread(SelectionStrategy strategy, vector<float>& best_scores_out, int run_seed, float* cpu_time, float* gpu_latency, Individual* global_best_individual, vector<float>& global_best_scores) {
    
    // Initialize the thread-local random number generator using the provided seed
    default_random_engine generator(run_seed); 

    // 1. Initialize Population
    vector<Individual> population;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population.push_back(create_random_individual(generator));
    }
    float gpu_time;
    // --- CRITICAL PRE-LOOP STEP: Initial Cost Evaluation ---
    // vector<float> initial_costs = evaluate_children_gpu(population, &gpu_time);
    vector<float> initial_costs = evaluate_children_gpu_heavy(population, &gpu_time);
    for (size_t i = 0; i < population.size(); ++i) {
        population[i].cost = initial_costs[i];
    }
    
    // Sort the population initially to find the best score
    sort(population.begin(), population.end(), compareIndividuals);
    best_scores_out.push_back(population.front().cost); // Store Gen 0 score
    vector<double> cpu_execution;
    vector<float> gpu_execution;
        // Global-best tracking
    float global_best_cost = population.front().cost;
    // global_best_scores.push_back(global_best_cost);
    if (global_best_individual)
      *global_best_individual = population.front();

    for (int gen = 0; gen < NUM_GEN; ++gen) {
        double start = get_time_ms();
        // ---------------- CPU Phase 1: threaded GA operators ----------------
        vector<Individual> children_pool(POPULATION_SIZE);

        // const int NUM_THREADS = 16;
        pthread_t threads[NUM_THREADS];
        ThreadArgs targs[NUM_THREADS];

        for (int t = 0; t < NUM_THREADS; t++) {
            targs[t].tid = t;
            targs[t].num_threads = NUM_THREADS;
            targs[t].run_seed = run_seed + gen * 123456;
            targs[t].strategy = strategy;
            targs[t].population = &population;
            targs[t].children_pool = &children_pool;

            pthread_create(&threads[t], nullptr, ga_cpu_thread, &targs[t]);
        }

        // Wait for all threads
        for (int t = 0; t < NUM_THREADS; t++)
            pthread_join(threads[t], nullptr);
        
        double end = get_time_ms();
        // cout << "CPU time (multithreads): " << end - start << " ms." << endl;
        cpu_execution.push_back(end-start);
        // --- CPU/GPU INTERFACE: Cost Evaluation Call (GPU SIMULATED) ---
        // vector<float> costs = evaluate_children_gpu(children_pool, &gpu_time); 
        vector<float> costs = evaluate_children_gpu_heavy(children_pool, &gpu_time); 
        gpu_execution.push_back(gpu_time);


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

                // --- Global best update (monotonic) ---
        if (population.front().cost > global_best_cost) {
            global_best_cost   = population.front().cost;
            if (global_best_individual)
                *global_best_individual = population.front();
        }

        // Store global best cost for convergence curve
        global_best_scores.push_back(global_best_cost); 
    }
    float tmp = 0;
    for(auto& a : cpu_execution){
        tmp += a;
    }
    tmp /= NUM_GEN;
    *cpu_time = tmp;
    tmp = 0;
    for(auto& b : gpu_execution){
        tmp += b;
    }
    tmp /= NUM_GEN;
    *gpu_latency = tmp;
}

/**
 * @function run_ga_wrapper
 * @brief Required wrapper function for pthread_create.
 * @param args_ptr Pointer to the ThreadArgs structure.
 */


int main(int argc, char* argv[]) {
    double start = get_time_ms();
    vector<float> sorted_results;
    vector<float> random_results;
    vector<float> global_best_sorted_results;
    vector<float> global_best_random_results;
    Individual global_best_sorted_individual;
    Individual global_best_random_individual;

    cout << "Running GA sequentially (" << NUM_GEN << " generations each)...\n";

    float sorted_cpu_time, random_cpu_time;
    float sorted_gpu_time, random_gpu_time;

    // Run Sorted Truncation GA
    int seed_sorted = (int)time(0);
    if(argv[1][0] == 's'){
        run_ga_singlethread(SelectionStrategy::SORTED_TRUNCATION, sorted_results, seed_sorted, &sorted_cpu_time, &sorted_gpu_time, &global_best_sorted_individual, global_best_sorted_results);
    }
    else if(argv[1][0] == 'm'){
        run_ga_multithread(SelectionStrategy::SORTED_TRUNCATION, sorted_results, seed_sorted, &sorted_cpu_time, &sorted_gpu_time, &global_best_sorted_individual, global_best_sorted_results);
    }


    // Run Pure Random GA
    int seed_random = (int)time(0) + 1;
    if(argv[1][0] == 's'){
        run_ga_singlethread(SelectionStrategy::RANDOM, random_results, seed_random, &random_cpu_time, &random_gpu_time, &global_best_random_individual, global_best_random_results);
    }
    else if(argv[1][0] == 'm'){
        run_ga_multithread(SelectionStrategy::RANDOM, random_results, seed_random, &random_cpu_time, &random_gpu_time, &global_best_random_individual, global_best_random_results);
    }
    float global_best_sorted_cost = global_best_sorted_results.back();
    float global_best_random_cost = global_best_random_results.back();
    // --- Final Comparison Report ---
    cout << "\n========================================================================\n";
    if(argv[1][0] == 's'){
        cout << "                           GA COMPARISON (SINGLE THREAD)                   \n";
    }
    else if(argv[1][0] == 'm'){
        cout << "                           GA COMPARISON (MULTI THREADS)                   \n";
    }
    cout << "========================================================================\n";
    cout << "    Generation  | Sorted Truncation Best Cost | Pure Random Best Cost\n";
    cout << "----------------|-----------------------------|---------------------\n";

    size_t num_generations = min(sorted_results.size(), random_results.size());
    
    bool find_sorted_optimal = false;
    bool find_random_optimal = false;

    for (size_t i = 0; i < num_generations; ++i) {
        if(i > 0) continue;
        
        if(i == 0)
          cout << setw(15) << "init" << " | ";
        else
          cout << setw(15) << i << " | ";
        
        cout << setw(27) << sorted_results[i] << " | ";
        cout << setw(19) << random_results[i] << endl;

        // if (sorted_results[i] == CHROMOSOME_LENGTH) {
        if(!find_sorted_optimal){
          if (sorted_results[i] == global_best_sorted_cost){
            //   if (i < num_generations) {
            //       cout << setw(45) << "--- OPTIMUM REACHED (Sorted) ---\n";
            //   }
              find_sorted_optimal = true;
              // break;
          }
        }
        if(!find_sorted_optimal){
          if (random_results[i] == global_best_random_cost){
            //   if (i < num_generations) {
            //       cout << setw(64) << "--- OPTIMUM REACHED (Random) ---\n";
            //   }
              find_random_optimal = true;
              // break;
          }
        }
    }

    // cout << "========================================================================\n";
    cout << setw(15) << "Best Score" << " | ";
    cout << setw(27) << global_best_sorted_cost << " | ";
    cout << setw(19) << global_best_random_cost << endl;
    //      << global_best_sorted_cost << " / " << CHROMOSOME_LENGTH << endl;
    // cout << "Final Best Score (Pure Random):       " 
    //      << global_best_random_cost << " / " << CHROMOSOME_LENGTH << endl;
    cout << "========================================================================\n";

    cout << setw(15) << "Crossover Time" << " | ";
    cout << setw(27) << sorted_cpu_time << " | ";
    cout << setw(19) << random_cpu_time << endl;

    cout << setw(15) << "Cost Time" << " | ";
    cout << setw(27) << sorted_gpu_time << " | ";
    cout << setw(19) << random_gpu_time << endl;



    cout << "========================================================================\n";
    // cout << "Avg. CPU Execution Time (Sorted Truncation): " 
    //      << sorted_cpu_time << " ms " << endl;
    // cout << "Avg. CPU Execution Time (Pure Random):       " 
    //      << random_cpu_time << " ms " << endl;
    // cout << "========================================================================\n";
    // cout << "Avg. Cost Execution Time (Sorted Truncation): " 
    //      << sorted_gpu_time << " ms " << endl;
    // cout << "Avg. Cost Execution Time (Pure Random):       " 
    //      << random_gpu_time << " ms " << endl;
    // cout << "========================================================================\n";

    // double end = get_time_ms();
    // cout << "Global Execution Time: " << end - start << " ms " << endl;
    // cout << "========================================================================\n";
    
    {
        string mode = (argv[1][0] == 's') ? "single" : "multi";
        string filename = "ga_convergence_cpu_" + mode + ".csv";
    
        ofstream csv(filename);
        csv << "Generation,Sorted,Random\n";
    
        size_t gens = min(global_best_sorted_results.size(), global_best_random_results.size());
        for (size_t i = 0; i < gens; i++) {
            csv << i << ","
                << global_best_sorted_results[i] << ","
                << global_best_random_results[i] << "\n";
        }
        csv.close();
    
        cout << "Saved convergence curve to " << filename << "\n";
    }

    // // --- Final Comparison Report ---
    // cout << "\n========================================================================\n";
    // if(argv[1][0] == 's'){
    //     cout << "                           GA COMPARISON (SINGLE THREAD)                   \n";
    // }
    // else if(argv[1][0] == 'm'){
    //     cout << "                           GA COMPARISON (MULTI THREADS)                   \n";
    // }
    // cout << "========================================================================\n";
    // cout << "Generation | Sorted Truncation Best Cost | Pure Random Best Cost\n";
    // cout << "-----------|-----------------------------|---------------------\n";

    // size_t num_generations = min(sorted_results.size(), random_results.size());

    // for (size_t i = 0; i < num_generations; ++i) {
    //     if(i > 0 && i < num_generations - 1) continue;

    //     cout << setw(10) << i << " | "
    //          << setw(27) << sorted_results[i] << " | "
    //          << setw(19) << random_results[i] << endl;

    //     if (sorted_results[i] >= CHROMOSOME_LENGTH) {
    //         if (i < num_generations - 1) {
    //             cout << "--- OPTIMUM REACHED (Sorted) ---\n";
    //         }
    //         break;
    //     }
    // }

    // cout << "========================================================================\n";
    // cout << "Final Best Score (Sorted Truncation): " 
    //      << sorted_results.back() << " / " << CHROMOSOME_LENGTH << endl;
    // cout << "Final Best Score (Pure Random):       " 
    //      << random_results.back() << " / " << CHROMOSOME_LENGTH << endl;

    // cout << "========================================================================\n";
    // cout << "Avg. CPU Execution Time (Sorted Truncation): " 
    //      << sorted_cpu_time << " ms " << endl;
    // cout << "Avg. CPU Execution Time (Pure Random):       " 
    //      << random_cpu_time << " ms " << endl;
    // cout << "========================================================================\n";
    // cout << "Avg. Cost Function Computation Time (Sorted Truncation): " 
    //      << sorted_gpu_time << " ms " << endl;
    // cout << "Avg. Cost Function Computation Time (Pure Random):       " 
    //      << random_gpu_time << " ms " << endl;
    // cout << "========================================================================\n";

    // double end = get_time_ms();
    // cout << "Global Execution Time: " << end - start << " ms " << endl;
    // cout << "========================================================================\n";
    return 0;
}




