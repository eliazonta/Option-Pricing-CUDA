LD_FLAGS_50 = -L/opt/cuda-5.0/lib64 -lcudart -lcufft
LD_FLAGS_NEW = -L/usr/local/cuda/lib64 -lcudart -lcufft
OBJS = option.o parameters.o utils.o

all: option.o parameters.o utils.o
	g++ -o option $(OBJS) $(LD_FLAGS_NEW)

# Assume that the CUDA version is really old and doesn't
# support floats.
old: option_old.o parameters.o utils.o
	g++ -o option $(OBJS) $(LD_FLAGS_50)

# Compile without an NVidia card using Ocelot
intel: option_ocelot.o parameters.o utils.o
	g++ -o option $(OBJS) `OcelotConfig -l` -lcufft

utils.o: utils.cu
	nvcc -g -c utils.cu

parameters.o: parameters.cpp
	gcc -g -c parameters.cpp

option.o: option.cu
	nvcc -g -c option.cu

option_old.o: option.cu
	nvcc -g -c option.cu -D USE_FLOAT

option_ocelot.o: option.cu
	nvcc -g -c option.cu -arch=sm_20

clean:
	rm -rf *.o option
