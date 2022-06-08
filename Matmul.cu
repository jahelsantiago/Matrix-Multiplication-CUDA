// includes
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCKS 32
#define THREADS 256

struct Matrix
{
    int rows;
    int cols;
    float **data;
};

/**
 * @brief create a function that returns a matrix with the numbers a to (n^2) + a
 * @param rows size of the matrix
 * @param cols size of the matrix
 * @param start start value of the matrix
 * @return a matrix with the numbers a to (n^2) + a
 */

Matrix *createMatrix(int rows, int cols, int start, float factor)
{
    // allocates memory for the matrix
    float **matrix = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        matrix[i] = (float *)malloc(cols * sizeof(float));
    }

    // fill the matrix with the numbers a to (n^2) + a
    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            matrix[i][j] = (start + (i * cols) + j) * factor;
        }
    }

    Matrix *matrix_struct = (Matrix *)malloc(sizeof(Matrix));
    matrix_struct->rows = rows;
    matrix_struct->cols = cols;
    matrix_struct->data = matrix;

    return matrix_struct;
}

/**
 * @brief print the matrix
 * @param matrix pointer to the matrix
 */
void printMatrix(Matrix *matrix)
{
    int i, j;
    for (i = 0; i < matrix->rows; i++)
    {
        for (j = 0; j < matrix->cols; j++)
        {
            printf("%f ", matrix->data[i][j]);
        }
        printf("\n");
    }
}

/**
 * @brief free the memory of the matrix
 * @param rows size of the matrix
 * @param matrix pointer to the matrix
 */
void freeMatrix(Matrix *matrix, bool isCuda)
{
    int i;
    for (i = 0; i < matrix->rows; i++)
    {
        if (isCuda)
        {
            cudaFree(matrix->data[i]);
        }
        else
        {
            free(matrix->data[i]);
        }
    }
    if (isCuda)
    {
        cudaFree(matrix->data);
    }
    else
    {
        free(matrix->data);
    }
    free(matrix);
}

Matrix *matrixMultiplication(Matrix *matrix_a, Matrix *matrix_b)
{
    int i, j, k;
    Matrix *matrix_c = (Matrix *)malloc(sizeof(Matrix));
    matrix_c->rows = matrix_a->rows;
    matrix_c->cols = matrix_b->cols;
    matrix_c->data = (float **)malloc(matrix_c->rows * sizeof(float *));
    for (i = 0; i < matrix_c->rows; i++)
    {
        matrix_c->data[i] = (float *)malloc(matrix_c->cols * sizeof(float));
    }
    for (i = 0; i < matrix_c->rows; i++)
    {
        for (j = 0; j < matrix_c->cols; j++)
        {
            matrix_c->data[i][j] = 0;
            for (k = 0; k < matrix_a->cols; k++)
            {
                matrix_c->data[i][j] += matrix_a->data[i][k] * matrix_b->data[k][j];
            }
        }
    }
    return matrix_c;
}

__global__ void matrixMultiplicationKernel(Matrix *matrix_a, Matrix *matrix_b, Matrix *matrix_c)
{
    // get the thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // get the total number of threads
    int total_threads = matrix_c->rows * matrix_c->cols;

    for (int i = id * total_threads; i < total_threads * id + total_threads; i++)
    {
        int row = i / matrix_c->cols;
        int col = i % matrix_c->cols;
        matrix_c->data[row][col] += matrix_a->data[row][0] * matrix_b->data[0][col];
    }
}

    Matrix *allocateMatrixCuda(Matrix * matrix)
    {
        Matrix cudaMatrix = {matrix->rows, matrix->cols, NULL};
        cudaMalloc((void **)&cudaMatrix.data, matrix->rows * matrix->cols * sizeof(float));
        return &cudaMatrix;
    }

    Matrix *parallelMatrixMultiplication(Matrix * matrix_a, Matrix * matrix_b)
    {

        // allocate memory for the result
        Matrix *matrix_c = createMatrix(matrix_a->rows, matrix_b->cols, 0, 0);

        // allocate memory on device
        Matrix *matrix_a_d = allocateMatrixCuda(matrix_a);
        Matrix *matrix_b_d = allocateMatrixCuda(matrix_b);
        Matrix *matrix_c_d = allocateMatrixCuda(matrix_c);

        // copy data to device
        cudaMemcpy(matrix_a_d->data, matrix_a->data, matrix_a->rows * matrix_a->cols * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(matrix_b_d->data, matrix_b->data, matrix_b->rows * matrix_b->cols * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(matrix_c_d->data, matrix_c->data, matrix_c->rows * matrix_c->cols * sizeof(float), cudaMemcpyHostToDevice);

        // total number of multiplication
        int total_multiplication = matrix_a->rows * matrix_b->cols * matrix_b->rows;
        int total_threads = BLOCKS * THREADS;

        // create the kernel
        matrixMultiplicationKernel<<<BLOCKS, THREADS>>>(matrix_a_d, matrix_b_d, matrix_c_d);

        // copy data from device to host
        cudaMemcpy(matrix_c->data, matrix_c_d->data, matrix_c->rows * matrix_c->cols * sizeof(float), cudaMemcpyDeviceToHost);

        // free memory
        freeMatrix(matrix_a_d, true);
        freeMatrix(matrix_b_d, true);
        freeMatrix(matrix_c_d, true);

        return matrix_c;
    }

    int main(int argc, char *argv[])
    {
        // create a matrix of size 3x3
        int a_rows = 3;
        int a_cols = 4;
        int a_start = 1;
        Matrix *matrix_a = createMatrix(a_rows, a_cols, a_start, 1);

        // create a matrix of size 3x3
        int b_rows = 3;
        int b_cols = 4;
        int b_start = 1;
        Matrix *matrix_b = createMatrix(b_rows, b_cols, b_start, 1);

        // mutiply the matrices
        Matrix *matrix_c = parallelMatrixMultiplication(matrix_a, matrix_b);

        // print the result
        printMatrix(matrix_c);

        // free the memory
        freeMatrix(matrix_a, false);
        freeMatrix(matrix_b, false);

        return 0;
    }