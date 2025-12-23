clean:
	rm ./teste

build:
	nvcc main.cu includes/*.cpp -o teste

run:
	rm -f ./teste; nvcc main.cu includes/*.cpp -o teste && clear && ./teste

runDebug:
	rm -f ./teste; rm -f ./trace.nsys-rep; nvcc -lineinfo -use_fast_math -O3 main.cu includes/*.cpp -o teste && clear; nsys profile -o trace ./teste;

openDebug:
	nsys-ui trace.nsys-rep