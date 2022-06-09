#!/bin/bash
> resultsOmp.txt
touch resultsOmp.txt
for matrixSize in 8 16 32 64 128 256 512 1024
do
echo "------------------------------------------------------" >> resultsOmp.txt
echo "Running test with $matrixSize x $matrixSize matrix"
for thread in 1 2 4 8 16
do
echo "Running test with $thread threads for a matrix of size $matrixSize x $matrixSize"
echo -e "\n>> $thread threads for a matrix of size $matrixSize x $matrixSize " >> resultsOmp.txt
./mmomp "${matrixSize}" "${matrixSize}" "${thread}" >> resultsOmp.txt
done
done