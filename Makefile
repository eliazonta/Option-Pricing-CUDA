all: option.o parameters.o
	g++ -o option option.o parameters.o `OcelotConfig -l` -lcufft

gpu: option.o parameters.o
	g++ -o option option.o parameters.o -lcufft

option.o: option.cu
	nvcc -g -c option.cu -arch=sm_20

parameters.o: parameters.cpp
	gcc -g -c parameters.cpp

clean:
	rm -rf *.o option
