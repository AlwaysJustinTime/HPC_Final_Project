# environment
SM := 35

GCC := g++
NVCC := nvcc

# Remove function
RM = rm -f
 
# Specify opencv Installation
#opencvLocation = /usr/local/opencv
opencvLIB= `pkg-config --libs opencv`
opencvINC= `pkg-config --cflags opencv`

# Compiler flags:
# -g    debugging information
# -Wall turns on most compiler warnings
GENCODE_FLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)
LIB_FLAGS := -lcudadevrt -lcudart

NVCCFLAGS := -O3
GccFLAGS = -fopenmp -O3 

# The build target executable:
TARGET  = kmeans_cuda
TARGETS = $(TARGET)

all: build

debug: GccFLAGS += -DDEBUG -g -Wall
debug: NVCCFLAGS += -g -G
debug: all

build: $(TARGETS)


$(TARGET): src/main.cpp src/config.h
	$(GCC) $(GccFLAGS) $(opencvLIB) $(opencvINC) $< -o $@
	

%: src/%.cpp
	$(GCC) $(GccFLAGS) $(opencvLIB) $(opencvINC) $< -o $@ $(OPENCV_LINK)
	
clean:
	$(RM) $(TARGETS) src/*.o *.o *.tar* *.core* *.out
