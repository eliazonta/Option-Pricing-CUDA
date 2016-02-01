all: option.o parameters.o
	g++ -o option option.o parameters.o `OcelotConfig -l` -lcufft

option.o:
	nvcc -c option.cu -arch=sm_20

parameters.o:
	gcc -c parameters.cpp

clean:
	rm -rf *.o option
