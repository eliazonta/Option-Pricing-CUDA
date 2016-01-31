all: option.o
	g++ -o option option.o `OcelotConfig -l`

option.o:
	nvcc -c option.cu -arch=sm_20

clean:
	rm -rf *.o option
