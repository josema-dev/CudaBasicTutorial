#include <cstdio>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "utils.h"

// m: number of rows of matrix A nad C
// n: number of columns matrix B and C
// k1: number of columns of A and rows B
void MatrixMulHost(float *A, float *B, float *C, const size_t rowsA, const size_t colsA, const size_t colsB)
{
	for (size_t i = 0; i < rowsA; i++)
	{
		for (size_t j = 0; j < colsB; j++)
		{
			for (size_t k = 0; k < colsA; k++)
			{
				size_t b_idx = k * colsB + j;
				size_t a_idx = i * colsA + k;
				size_t c_idx = i * colsB + j;
				C[c_idx] += B[b_idx] * A[a_idx];
				printf("a[%d] = %f, b[%d] = %f, c[%d] = %f\n", a_idx, A[a_idx], b_idx, B[b_idx], c_idx, C[c_idx]);
			}
		}
	}
}
template<int BLOCK_SIZE>
__global__ void MatrixMul(float *A, float *B, float *C, const size_t rowsA, const size_t colsA, const size_t colsB)
{

}

int main()
{
	constexpr size_t MATRIX_ROW_A = 3;
	constexpr size_t MATRIX_COL_A = 2;
	constexpr size_t MATRIX_ROW_B = MATRIX_COL_A;
	constexpr size_t MATRIX_COL_B = 4;
	constexpr size_t MATRIX_ROW_C = MATRIX_ROW_A;
	constexpr size_t MATRIX_COL_C = MATRIX_COL_B;

	float *aMatrixHost, *bMatrixHost, *cMatrixHost;

	aMatrixHost = (float*)malloc(sizeof(float) * MATRIX_ROW_A * MATRIX_COL_A);
	bMatrixHost = (float*)malloc(sizeof(float) * MATRIX_ROW_B * MATRIX_COL_B);
	cMatrixHost = (float*)malloc(sizeof(float) * MATRIX_ROW_C * MATRIX_COL_C);

	InitMatrixA(aMatrixHost, MATRIX_COL_A * MATRIX_ROW_A);
	InitMatrixB(bMatrixHost, MATRIX_COL_B * MATRIX_ROW_B);
	InitMatrixC(cMatrixHost, MATRIX_COL_C * MATRIX_ROW_C);

	MatrixMulHost(aMatrixHost, bMatrixHost, cMatrixHost, MATRIX_ROW_A, MATRIX_COL_A, MATRIX_COL_B);


	return 0;
}