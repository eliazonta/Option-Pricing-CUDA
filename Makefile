ifneq ($(wildcard /usr/local/cuda/.*),)
	LD_CUDA = -L/usr/local/cuda/lib64
else
	LD_CUDA = -L/opt/cuda-5.0/lib64
endif

LD_FLAGS = $(LD_CUDA) -lcudart -lcufft -lfftw3 -lfftw3f
OBJS = parameters.o utils.o fftwproxy.o

all: option.o $(OBJS)
	g++ -o option option.o $(OBJS) $(LD_FLAGS)

# Compile the program to work in single-precision.
# This is needed if the CUDA version is really old and doesn't
# support double-precision.
float: option_float.o $(OBJS)
	g++ -o option option.o $(OBJS) $(LD_FLAGS)

# Compile without an NVidia card using Ocelot
intel: option_ocelot.o $(OBJS)
	g++ -o option option_ocelot.o $(OBJS) `OcelotConfig -l` -lcufft

utils.o: utils.cu
	nvcc -g -c -std=c++11 utils.cu

parameters.o: parameters.cpp
	gcc -g -c parameters.cpp

fftwproxy.o: fftwproxy.cpp
	gcc -g -c fftwproxy.cpp

option.o: option.cu
	nvcc -g -c -std=c++11 option.cu

option_float.o: option.cu
	nvcc -g -c -std=c++11 option.cu -D USE_FLOAT

option_ocelot.o: option.cu
	nvcc -g -c -std=c++11 option.cu -arch=sm_20

clean:
	rm -rf *.o option
