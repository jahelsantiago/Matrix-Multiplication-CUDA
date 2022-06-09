#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


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
struct Matrix *createMatrix(int rows, int cols, int start, float factor)
{
    // allocates memory for the matrix
    float **matrix = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        matrix[i] = (float *)malloc(cols * sizeof(float));
    }

    // fill the matrix with the numbers to (n^2) + a
    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            matrix[i][j] = (start + (i * cols) + j) * factor;
        }
    }

    struct Matrix *matrix_struct = (struct Matrix *)malloc(sizeof(struct Matrix));
    matrix_struct->rows = rows;
    matrix_struct->cols = cols;
    matrix_struct->data = matrix;

    return matrix_struct;
}

/**
 * @brief multiply the matrices matrix_a and matrix_b and store the result in matrix_c
 * @param matrix_a pointer to the first matrix
 * @param matrix_b pointer to the second matrix
 * @param matrix_c pointer to the result matrix
 * @param rows_a number of rows of the first matrix
 * @param cols_a number of columns of the first matrix
 * @param rows_b number of rows of the second matrix
 * @param cols_b number of columns of the second matrix
 * @return
 */
struct Matrix *matrixMultiplicationOmp(struct Matrix *matrix_a, struct Matrix *matrix_b, struct Matrix *matrix_c, int rows_a, int cols_a, int rows_b, int cols_b, int threads)
{
    int i, j, k;
    omp_set_num_threads(threads);

    #pragma omp parallel for private(i, j, k), shared(matrix_a, matrix_b, matrix_c)
    for (i = 0; i < rows_a; i++)
    {
        for (j = 0; j < cols_b; j++)
        {
            for (k = 0; k < cols_a; k++)
            {
                matrix_c->data[i][j] += matrix_a->data[i][k] * matrix_b->data[k][j];
            }
        }
    }
    return matrix_c;
}

/**
 * @brief allocate memory for the matrix and create a matrix for result. With this calls the matrix multiplication function
 * @param matrixA pointer to the first matrix
 * @param matrixB pointer to the second matrix
 * @return the result of the multiplication between matrixA and matrixB as a new matrix
 */
struct Matrix *sequentialMultiplication(struct Matrix *matrixA, struct Matrix *matrixB, int threads) {
    // check if the matrices can be multiplied
    if (matrixA->cols != matrixB->rows) {
        printf("Error: The matrices cannot be multiplied.\n");
        return NULL;
    }

    // allocate memory for the result matrix
    struct Matrix *matrixC = createMatrix(matrixA->rows, matrixB->cols, 0, 0);

    // get the result of the multiplication between matrixA and matrixB
    struct Matrix *result = matrixMultiplicationOmp(matrixA, matrixB, matrixC, matrixA->rows, matrixA->cols, matrixB->rows, matrixB->cols, threads);

    return result;
}

/**
 * @brief print the matrix
 * @param matrix pointer to the matrix
 */
void printMatrix(struct Matrix *matrix)
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
void freeMatrix(struct Matrix *matrix)
{
    for (int i = 0; i < matrix->rows; i++)
    {
        free(matrix->data[i]);
    }
    free(matrix->data);
    free(matrix);
}

int main(int argc, char** argv)
{

    if (argc != 4)
    {
        printf("Usage: %s <rows> <cols> <threads> \n", argv[0]);
        return 1;
    }

    int num_rows = atoi(argv[1]);
    int num_cols = atoi(argv[2]);
    int num_threads = atoi(argv[3]);

    // create the matrices
    struct Matrix *matrix_a = createMatrix(num_rows, num_cols, 1, 1);
    struct Matrix *matrix_b = createMatrix(num_rows, num_cols, 1, 1);

    // multiply the matrices and take the time
    clock_t start = clock();
    struct Matrix *matrix_c = sequentialMultiplication(matrix_a, matrix_b, num_threads);
    clock_t end = clock();

    // print the time
    printf("Time: %.15Lf\n", (long double)(end - start) / CLOCKS_PER_SEC);

    /*printf("Matrix A:\n");
    printMatrix(matrix_a);
    printf("Matrix B:\n");
    printMatrix(matrix_b);
    printf("Matrix C:\n");
    printMatrix(matrix_c);*/

    // free the memory of the matrices
    freeMatrix(matrix_a);
    freeMatrix(matrix_b);
    freeMatrix(matrix_c);

    return 0;
}
