ifneq ($(wildcard /usr/local/cuda/.*),)
	LD_CUDA = -L/usr/local/cuda/lib64
else
	LD_CUDA = -L/opt/cuda-5.0/lib64
endif

LD_FLAGS = $(LD_CUDA) -lcudart -lcufft -lfftw3 -lfftw3f
OBJS = params.o utils.o fftwproxy.o

# compilers and version
CC := gcc
CXX := g++
NVCC := nvcc
VERSION := -std=c++11

# folders
SRC := src
OBJ := obj
BIN := bin
UTIL := utils

all: $(OBJ)/option.o $(OBJ)/$(OBJS)
	$(CXX) -o $(BIN)/option $(OBJ)/option.o $(OBJS) $(LD_FLAGS)

# Compile the program to work in single-precision.
# This is needed if the CUDA version is really old and doesn't
# support double-precision.
float: $(OBJ)/option_float.o $(OBJS)
	$(CXX) -o $(BIN)/option $(OBJ)/option.o $(OBJS) $(LD_FLAGS)

$(OBJ)/utils.o: $(SRC)/$(UTIL)/utils.cu
	$(NVCC) -g -c $(VERSION) $(SRC)/$(UTIL)/utils.cu

$(OBJS)/params.o: $(SRC)/$(UTIL)/params.cpp
	$(CC) -g -c $(SRC)/$(UTIL)/params.cpp

$(SRC)/fftwproxy.o: $(SRC)/fftwproxy.cpp
	$(CC) -g -c $(SRC)/fftwproxy.cpp

$(OBJ)/option.o: $(SRC)/option.cu
	$(NVCC) -g -c $(VERSION) $(SRC)/option.cu

$(OBJ)/option_float.o: $(SRC)/option.cu
	$(NVCC) -g -c $(VERSION) $(SRC)/option.cu -D USE_FLOAT

$(OBJ)/option_ocelot.o: $(SRC)/option.cu
	$(NVCC) -g -c $(VERSION) $(SRC)/option.cu -arch=sm_20

clean:
	rm -rf  option bin obj *.o *.out *.err 
