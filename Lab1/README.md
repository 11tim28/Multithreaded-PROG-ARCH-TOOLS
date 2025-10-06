# Multithreaded PROG/ARCH/TOOLS Lab 1

## Parallel Processing Pipeline with Semaphores
This project implements a 3 stage pipeline with bounded queues to perform genetic algorithm. 

## Project Structure
Genetic algorithm consists of parent individuals selection, crossover operation and cost function calculation. In this project, I define the 3 stage pipeline with the following structure:
- **Selection**: Select two parent individuals from parents' pool, then push them to the bounded queue.
- **Crossover**: Pop two individuals from the queue, perform crossover operation and produce two offsprings. Finally, push them to the bounded queue for next stage.
- **Cost**: Pop two individuals from the queue, calculate their cost function value, and then add them to the offspring pool.
Initially, there are 100 parents in the parents' pool. User can specify the number of offsprings they want to produce and the number of threads in each stage. For parameter setting, refer to the next section.

## How to run?
The code design is in ga.cpp. To compile it, simply "make" as a Makefile is provided.
'''bash
#compile
make
'''


