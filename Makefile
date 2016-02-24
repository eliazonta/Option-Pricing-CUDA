LD_FLAGS = -L/opt/cuda-5.0/lib64 -lcudart -lcufft
#LD_FLAGS = -L/usr/local/cuda/lib64 -lcudart -lcufft

all: option.o parameters.o utils.o
	g++ -o option option.o parameters.o utils.o $(LD_FLAGS)

# Compile without an NVidia card using Ocelot
intel: option_ocelot.o parameters.o utils.o
	g++ -o option option.o parameters.o utils.o `OcelotConfig -l` -lcufft

option.o: option.cu
	nvcc -g -c option.cu

utils.o: utils.cu
	nvcc -g -c utils.cu

option_ocelot.o: option.cu
	nvcc -g -c option.cu -arch=sm_20

parameters.o: parameters.cpp
	gcc -g -c parameters.cpp

clean:
	rm -rf *.o option
