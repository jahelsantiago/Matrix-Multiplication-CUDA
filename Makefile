SHELL := /bin/bash

compile:
	gcc mtrxMultOMP.c -o mmomp -lm -fopenmp
	nvcc Matmul.cu -o mmcuda
omp:
	sh testOmp.sh
cuda:
	sh testCuda.sh