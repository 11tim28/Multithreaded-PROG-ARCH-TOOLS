# Multithreaded PROG/ARCH/TOOLS Lab 1

## Parallel Processing Pipeline with Semaphores
This project implements a 3 stage pipeline with bounded queues to perform genetic algorithm. 

## Project Structure
Genetic algorithm consists of parent individuals selection, crossover operation and cost function calculation. In this project, I define the 3 stage pipeline with the following structure:
- **Selection stage**: Select two parent individuals from parents' pool, then push them to the bounded queue.
- **Crossover stage**: Pop two individuals from the queue, perform crossover operation and produce two offsprings. Finally, push them to the bounded queue for next stage.
- **Cost stage**: Pop two individuals from the queue, calculate their cost function value, and then add them to the offspring pool.

Initially, there are 100 parents in the parents' pool. User can specify the number of offsprings they want to produce and the number of threads in each stage. For parameter setting, refer to the next section.

## How to run?
The code design is in ga.cpp. To compile it, simply "make" as a Makefile is provided.
```bash
# compile
make
```
After compiling, user can run the program using the following command
```bash
# Run the code
./ga <#Selection Threads> <#Crossover Threads> <#Cost Threads> <number_of_offsprings>
```
For example, if user want to produce 16 offsprings, with 4 threads in each stage
```bash
# 16 offsprings, 4 threads in each stage
./ga 4 4 4 16
```

## Experiments Results
The following table shows the latency of different number of offsprings and number of threads. The following data are measure multiple times and take average from them. For simplicity, I use the same number of threads in each stage. 
| Number of Offspring | 1 thread/stage | 2 threads/stage | 4 threads/stage | 8 threads/stage |
|---------------------|----------------|-----------------|-----------------|-----------------| 
| 16                  | 122.6 ms       | 96.2 ms         | 83.8 ms         | 76.4 ms         |
| 64                  | 188.5 ms       | 183.2 ms        | 104 ms          | 100 ms          |

In addition to latency, I also record the value of cost in offspring pool. Since I always select the parents with highest score in selection stage, I expect to get offsprings with higher score according to genetic algorithm. In my experiment, I observe the offspring pool has an approximate **25%** increase compared to parents pool, which verified the assumption of genetic algorithm.

## Environment
I test and run this project on LRC mario server.

## AI tools
I use UT Spark to generate testcase and test plan.
