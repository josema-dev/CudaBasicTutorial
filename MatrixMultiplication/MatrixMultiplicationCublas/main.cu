#include <cstdio>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

#include "utils.h"

// rowsA: number of rows of matrix A nad C
// colsA: number of columns matrix B and C
// k1: number of columns of A and rows B
void MatrixMulHost(float *A, float *B, float *C, const size_t rowsA, const size_t colsA, const size_t colsB)
{
	for (size_t i = 0; i < rowsA; i++)
	{
		for (size_t j = 0; j < colsB; j++)
		{
			float sum = 0;
			size_t c_idx = i * colsB + j;
			for (size_t k = 0; k < colsA; k++)
			{
				size_t b_idx = k * colsB + j;
				size_t a_idx = i * colsA + k;
				 sum += B[b_idx] * A[a_idx];
				//printf("a[%d] = %f, b[%d] = %f, c[%d] = %f\n", a_idx, A[a_idx], b_idx, B[b_idx], c_idx, C[c_idx]);
			}
			C[c_idx] = sum;
		}
	}
}

int main()
{
	constexpr size_t MATRIX_ROW_A = 640;
	constexpr size_t MATRIX_COL_A = 480;
	constexpr size_t MATRIX_ROW_B = MATRIX_COL_A;
	constexpr size_t MATRIX_COL_B = 320;
	constexpr size_t MATRIX_ROW_C = MATRIX_ROW_A;
	constexpr size_t MATRIX_COL_C = MATRIX_COL_B;

	float *aMatrixHost, *bMatrixHost, *cMatrixHost, *resMatHost;

	aMatrixHost = (float*)malloc(sizeof(float) * MATRIX_ROW_A * MATRIX_COL_A);
	bMatrixHost = (float*)malloc(sizeof(float) * MATRIX_ROW_B * MATRIX_COL_B);
	cMatrixHost = (float*)malloc(sizeof(float) * MATRIX_ROW_C * MATRIX_COL_C);
	resMatHost = (float*)malloc(sizeof(float) * MATRIX_ROW_C * MATRIX_COL_C);

	RandomInit(aMatrixHost, MATRIX_COL_A * MATRIX_ROW_A);
	RandomInit(bMatrixHost, MATRIX_COL_B * MATRIX_ROW_B);
	InitMatrixC(cMatrixHost, MATRIX_COL_C * MATRIX_ROW_C);

	MatrixMulHost(aMatrixHost, bMatrixHost, cMatrixHost, MATRIX_ROW_A, MATRIX_COL_A, MATRIX_COL_B);

	float *aMatDev, *bMatDev, *cMatDev;
	checkCudaErrors(cudaMalloc(&aMatDev, sizeof(float) * MATRIX_ROW_A * MATRIX_COL_A));
	checkCudaErrors(cudaMalloc(&bMatDev, sizeof(float) * MATRIX_ROW_B * MATRIX_COL_B));
	checkCudaErrors(cudaMalloc(&cMatDev, sizeof(float) * MATRIX_ROW_C * MATRIX_COL_C));

	checkCudaErrors(cudaMemcpy(aMatDev, aMatrixHost, sizeof(float) * MATRIX_ROW_A * MATRIX_COL_A, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(bMatDev, bMatrixHost, sizeof(float) * MATRIX_ROW_B * MATRIX_COL_B, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(cMatDev, 0, sizeof(float) * MATRIX_ROW_C * MATRIX_COL_C));
	
	const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasHandle_t handle;
	
	checkCudaErrors(cublasCreate(&handle));
	//col order swap a and b
	checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_COL_B, MATRIX_ROW_A, MATRIX_COL_A, &alpha, bMatDev, MATRIX_COL_B, aMatDev, MATRIX_COL_A, &beta, cMatDev, MATRIX_COL_C));
	checkCudaErrors(cudaMemcpy(resMatHost, cMatDev, sizeof(float) * MATRIX_ROW_C * MATRIX_COL_C, cudaMemcpyDeviceToHost));
	checkCudaErrors(cublasDestroy(handle));

	float maxDiff = CheckResults(cMatrixHost, resMatHost, MATRIX_ROW_C * MATRIX_COL_C);

	printf("Max difference between DEVICE and HOST: %f\n", maxDiff);

	// for (int i = 0; i < MATRIX_COL_C * MATRIX_ROW_C; i++)
	// {
	// 	printf("res[%d]: %f / %f \n", i, resMatHost[i], cMatrixHost[i]);
	// }

	checkCudaErrors(cudaFree(aMatDev));
	checkCudaErrors(cudaFree(bMatDev));
	checkCudaErrors(cudaFree(cMatDev));

	free(aMatrixHost);
	free(bMatrixHost);
	free(cMatrixHost);
	free(resMatHost);

	return 0;
}