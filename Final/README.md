# Multithreaded PROG/ARCH/TOOLS Final Project
## Genetic Algorithm on CPU/GPU Heterogeneous System
This project implements a simple Genetic Algorithm (GA) in two variants:

1. **CPU-only implementation** (pthread-based multithreading)  
2. **CPU + GPU hybrid implementation** (CUDA-based fitness evaluation)

The GPU version accelerates fitness computation — the most expensive part of the GA — using a CUDA kernel. The whole system is compiled and run on TACC.
## Project Structure
- **main_cpu.cpp**: pure CPU implementation, include single thread and multithreads.
- **main_gpu.cpp**: CPU + GPU implementation, include single thread and multithreads.
- **ga_types.h**: parameters and struct declaration
- **gpu_interface.h**: GPU kernel interface declaration
- **gpu_cost.cu**: CUDA kernel for cost function computation
- **Makefile**

## How to run?
After logging to TACC, first setup the environment.
```bash
idev -p rtx
```
Then, load gcc.
```bash
module load gcc/6.3.0
```
To this point, the environment is set. To run pure CPU mode:
```bash
# Pure CPU mode
make cpu
```
To run CPU + GPU mode:
```bash
# CPU + GPU mode
make gpu
```