LD_FLAGS_50 = -L/opt/cuda-5.0/lib64 -lcudart -lcufft
LD_FLAGS_NEW = -L/usr/local/cuda/lib64 -lcudart -lcufft -lfftw3
OBJS = parameters.o utils.o fftwproxy.o

all: option.o $(OBJS)
	g++ -o option option.o $(OBJS) $(LD_FLAGS_NEW)

# Assume that the CUDA version is really old and doesn't
# support floats.
old: option_old.o $(OBJS)
	g++ -o option option_old.o $(OBJS) $(LD_FLAGS_50)

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

option_old.o: option.cu
	nvcc -g -c -std=c++11 option.cu -D USE_FLOAT

option_ocelot.o: option.cu
	nvcc -g -c -std=c++11 option.cu -arch=sm_20

clean:
	rm -rf *.o option
