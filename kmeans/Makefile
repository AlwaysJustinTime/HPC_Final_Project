# environment
SM := 35

GCC := g++
NVCC := nvcc
MPICC := mpicc

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
GccFLAGS = -fopenmp -O3 -lm
MPIccFLAGS = -O3 -lm

# The build target executable:
TARGETMPI  = kmeans_mpi
TARGETOMP = kmeans_omp
TARGETS = $(TARGETMPI) $(TARGETOMP)

all: build

debug: GccFLAGS += -DDEBUG -g -Wall
debug: NVCCFLAGS += -g -G
debug: all

build: $(TARGETS)

$(TARGETMPI): src/mainMPI.cpp src/config.h
	$(MPICC) $(MPIccFLAGS) $(opencvLIB) $(opencvINC) -c $< -o $@

$(TARGETOMP): src/mainOMP.cpp src/config.h
	$(GCC) $(GccFLAGS) $(opencvLIB) $(opencvINC) -c $< -o $@

%: src/%.cpp
	$(GCC) $(GccFLAGS) $(opencvLIB) $(opencvINC) $< -o $@ $(OPENCV_LINK)
	
clean:
	$(RM) $(TARGETS) src/*.o *.o *.tar* *.core*
