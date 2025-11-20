# Compiler
NVCC = nvcc
CXXFLAGS = -std=c++14 -O2
LDFLAGS = -lpthread

# Target name
TARGET = ga_gpu

# Source files
SRCS = main.cpp gpu_cost.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
