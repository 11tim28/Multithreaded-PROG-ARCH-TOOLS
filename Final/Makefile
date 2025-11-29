# Compiler
NVCC = nvcc
CC = g++
CXXFLAGS = -std=c++14 -O2
LDFLAGS = -lpthread

# Target name
GPU_TARGET = ga_gpu
CPU_TARGET = ga_cpu

# Source files
GPU_SRCS = main_gpu.cpp gpu_cost.cu
CPU_SRCS = main_cpu.cpp

gpu: $(GPU_TARGET)

$(GPU_TARGET): $(GPU_SRCS)
	$(NVCC) $(CXXFLAGS) $(GPU_SRCS) -o $(GPU_TARGET) $(LDFLAGS)
	$(GPU_TARGET) singlethread
	$(GPU_TARGET) multithreads

cpu: $(CPU_TARGET)

$(CPU_TARGET): $(CPU_SRCS)
	$(CC) $(CPU_SRCS) -o $(CPU_TARGET) $(CXXFLAGS) $(LDFLAGS)
	$(CPU_TARGET) singlethread
	$(CPU_TARGET) multithreads

clean:
	rm -f $(CPU_TARGET) $(GPU_TARGET)
