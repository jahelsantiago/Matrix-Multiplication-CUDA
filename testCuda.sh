#!/bin/bash
> resultsCuda.txt
touch resultsCuda.txt
for matrixSize	in 8 16 32 64 128 256 512 1024
do
echo "------------------------------------------------------" >> resultsCuda.txt
echo "Running test with $matrixSize x $matrixSize matrix"
for blocks in 1 2 4 8 16
do
echo "Running test with $blocks threads"
for thread in 1 32 128 256 1024
do
echo "Running test with $blocks blocks and $thread threads per block for a matrix of size $matrixSize x $matrixSize"
echo -e "\n>> $blocks blocks and $thread threads per block for a matrix of size $matrixSize x $matrixSize " >> resultsCuda.txt
./mmcuda "${matrixSize}" "${matrixSize}" "${blocks}" "${thread}" >> resultsCuda.txt
done
done
done