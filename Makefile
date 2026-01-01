build:
	nvcc -std=c++17 -arch=native -O2 src/main.cu include/*.cpp -o bin/app -lcublas

buildDebug:
	nvcc -G -g -std=c++17 -arch=native -O2 src/main.cu include/*.cpp -o bin/app -lcublas

run:
	rm bin/app; make build; bin/app
