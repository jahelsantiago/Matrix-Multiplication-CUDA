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
 * @param start start value of the matrix numbers [start, start+ 1, ..., start + (n^2)]
 * @param factor factor to multiply the numbers factor * [start, start+ 1, ..., start + (n^2)]
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
 * @param name name of the matrix
 */
void printMatrix(Matrix *matrix, char *name)
{
    int i, j;
    printf("%s\n", name);
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

/**
 * @brief multiply two matrices
 * @param matrix_a pointer to the first matrix
 * @param matrix_b pointer to the second matrix
 * @param matrix_c pointer to the result matrix
 * @param rows_a number of rows of the first matrix
 * @param cols_a number of columns of the first matrix
 * @param rows_b number of rows of the second matrix
 * @param cols_b number of columns of the second matrix  
 * @return the result of the multiplication
 */
__global__ void matrixMultiplicationKernel(float *matrix_a, float *matrix_b, float *matrix_c, int rows_a, int cols_a, int rows_b, int cols_b)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = rows_a * cols_b * cols_a;

    for (int i = id * total_threads; i < total_threads * id + total_threads; i++)
    {
        //get the row and column of the matrix_a
        int id_2d = i % (rows_a * cols_a);

        int row_a = id_2d / cols_a;
        int col_a = id_2d % cols_a;

        //get the row and column of the matrix_b
        int row_b = col_a;
        int col_b = i / (rows_a * cols_a);

        //get the row and column of the matrix_c
        int row_c = row_a;
        int col_c = col_b;

        //get the index of the matrix_b
        int index_b = row_b * cols_b + col_b;

        //get the index of the matrix_c
        int index_c = row_c * cols_b + col_c;

        matrix_c[index_c] += matrix_a[row_a * cols_a + col_a] * matrix_b[index_b];
    }
}

/**
 * @brief Convert the matrix to a 1D array
 * 
 * @param matrix 
 * @return float* 
 */
float *flatMatrix(Matrix matrix)
{
    float *flat_matrix = (float *)malloc(matrix.rows * matrix.cols * sizeof(float));
    for (int i = 0; i < matrix.rows; i++)
    {
        for (int j = 0; j < matrix.cols; j++)
        {
            flat_matrix[i * matrix.cols + j] = matrix.data[i][j];
        }
    }
    return flat_matrix;
}


/**
 * @brief Convert the matrix to a 1D array and save it on the GPU
 * 
 * @param matrix 
 * @return float* 
 */
float *allocateMatrixCUDA(Matrix matrix)
{
    float *flat_matrix = flatMatrix(matrix);
    float *flat_matrix_cuda;
    cudaMalloc((void **)&flat_matrix_cuda, matrix.rows * matrix.cols * sizeof(float));
    cudaMemcpy(flat_matrix_cuda, flat_matrix, matrix.rows * matrix.cols * sizeof(float), cudaMemcpyHostToDevice);
    free(flat_matrix);
    return flat_matrix_cuda;
}

/**
 * @brief Convert the matrix to a 1D array and save it on the GPU
 * 
 * @param matrix 
 * @return float* 
 */
bool checkMatrixDimensions(Matrix *matrix_a, Matrix *matrix_b)
{
    //chek if matrix multiplication is possible 
    if (matrix_a->cols != matrix_b->rows)
    {
        printf("Matrix multiplication is not possible\n");
        return false;
    }
}

/**
 * @brief multiply two matrices
 * @param flat_matrix_a pointer to the first matrix
 * @param rows number of rows of the first matrix
 * @param cols number of columns of the first matrix
 * @return the result of the multiplication
 */
Matrix reshapeMatrix(float *flat_matrix, int rows, int cols)
{
    Matrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.data = (float **)malloc(matrix.rows * sizeof(float *));
    for (int i = 0; i < matrix.rows; i++)
    {
        matrix.data[i] = (float *)malloc(matrix.cols * sizeof(float));
    }
    for (int i = 0; i < matrix.rows; i++)
    {
        for (int j = 0; j < matrix.cols; j++)
        {
            matrix.data[i][j] = flat_matrix[i * matrix.cols + j];
        }
    }
    return matrix;
}

/**
 * @brief multiply two matrices
 * @param matrix_a pointer to the first matrix
 * @param matrix_b pointer to the second matrix
 * @return the result of the multiplication into an 1D array
 */
Matrix *parallelMatrixMultiplication(Matrix *matrix_a, Matrix *matrix_b)
{
    //check matrix dimensions and raise error if not equal
    if (!checkMatrixDimensions(matrix_a, matrix_b))
    {
        printf("Matrix dimensions do not match\n");
        return NULL;
    }

    // allocate memory for the result
    Matrix *matrix_c = createMatrix(matrix_a->rows, matrix_b->cols, 0, 0);
    float *flat_matrix_c = flatMatrix(*matrix_c);
    // allocate memory on GPU
    float *matrix_a_gpu = allocateMatrixCUDA(*matrix_a);
    float *matrix_b_gpu = allocateMatrixCUDA(*matrix_b);
    float *matrix_c_gpu = allocateMatrixCUDA(*matrix_c);

    // create the kernel
    matrixMultiplicationKernel<<<1, 1>>>(matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, matrix_a->rows, matrix_a->cols, matrix_b->rows, matrix_b->cols);

    // copy data from device to host
    cudaMemcpy(flat_matrix_c, matrix_c_gpu, matrix_c->rows * matrix_c->cols * sizeof(float), cudaMemcpyDeviceToHost);

    //reshape the matrix
    *matrix_c = reshapeMatrix(flat_matrix_c, matrix_c->rows, matrix_c->cols);

    // free memory
    cudaFree(matrix_a_gpu);
    cudaFree(matrix_b_gpu);
    cudaFree(matrix_c_gpu);

    return matrix_c;
}

int main(int argc, char *argv[])
{
    // create a matrix of size 3x3
    int a_rows = 4;
    int a_cols = 2;
    int a_start = 1;
    Matrix *matrix_a = createMatrix(a_rows, a_cols, a_start, 1);
    printMatrix(matrix_a, "Matrix A");

    // create a matrix of size 3x3
    int b_rows = 2;
    int b_cols = 9;
    int b_start = 1;
    Matrix *matrix_b = createMatrix(b_rows, b_cols, b_start, 1);
    printMatrix(matrix_b, "Matrix B");

    // mutiply the matrices
    Matrix *matrix_c = parallelMatrixMultiplication(matrix_a, matrix_b);
    printMatrix(matrix_c, "Matrix C");

    // free the memory
    freeMatrix(matrix_a, false);
    freeMatrix(matrix_b, false);

    return 0;
}