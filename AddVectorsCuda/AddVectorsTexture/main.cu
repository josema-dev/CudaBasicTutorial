#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "utils.h"

texture<float> texIn;

__global__ void AddVectors(float *a, float *c, const size_t elementNum)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx<elementNum)
		c[idx] = a[idx] + tex1Dfetch<float>(texIn, idx);
}

void AddVectorsHost(float *a, float *b, float *c, const size_t elementNum)
{
	for(size_t i=0; i<elementNum; i++)
	{
		c[i] = a[i] + b[i];
	}
}

int main()
{
	constexpr size_t VECTOR_SIZE = 1025;
	constexpr size_t THREADS_PER_BLOCK = 256;
	float *a_host, *b_host, *c_host, *res_host;
	float *a_dev, *b_dev, *c_dev;

	cudaEvent_t start, stop;

	// Alloc host data
	a_host = (float*)malloc(sizeof(float) * VECTOR_SIZE);
	b_host = (float*)malloc(sizeof(float) * VECTOR_SIZE);
	c_host = (float*)malloc(sizeof(float) * VECTOR_SIZE);
	res_host = (float*)malloc(sizeof(float) * VECTOR_SIZE);

	// Init host data
	InitData(a_host, b_host, c_host, VECTOR_SIZE);

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start));

	//Alloc device data
	checkCudaErrors(cudaMalloc(&a_dev, sizeof(float) * VECTOR_SIZE));
	checkCudaErrors(cudaMalloc(&b_dev, sizeof(float) * VECTOR_SIZE));
	checkCudaErrors(cudaMalloc(&c_dev, sizeof(float) * VECTOR_SIZE));

	//Texture memory binding
	checkCudaErrors(cudaBindTexture(NULL, texIn, b_dev, VECTOR_SIZE * sizeof(float)));

	// Copy host data to dev
	checkCudaErrors(cudaMemcpy(a_dev, a_host, sizeof(float) * VECTOR_SIZE, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(b_dev, b_host, sizeof(float) * VECTOR_SIZE, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(c_dev, 0, sizeof(float) * VECTOR_SIZE));

	//Compute device
	dim3 blocks(1,1,1);
	dim3 threads(1,1,1);

	threads.x = THREADS_PER_BLOCK;
	blocks.x = (VECTOR_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	AddVectors<<<blocks, threads>>>(a_dev, c_dev, VECTOR_SIZE);

	cudaError_t errorNum = cudaGetLastError();
	if(errorNum != cudaSuccess)
		printf("Cuda error running kernel.\n%s\n", cudaGetErrorName(errorNum));
	
	//Copy data back to host
	checkCudaErrors(cudaMemcpy(res_host, c_dev, sizeof(float) * VECTOR_SIZE, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaUnbindTexture(texIn));
	// Free device data
	checkCudaErrors(cudaFree(a_dev));
	checkCudaErrors(cudaFree(b_dev));
	checkCudaErrors(cudaFree(c_dev));

	checkCudaErrors(cudaEventRecord(stop));
	checkCudaErrors(cudaEventSynchronize(stop));
	float elapsedTime = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Cuda execution time: %f ms.\n", elapsedTime);

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	// Compute host
	AddVectorsHost(a_host, b_host, c_host, VECTOR_SIZE);

	//Check results
	float maxDiff = CheckResults(c_host, res_host, VECTOR_SIZE);

	printf("Max difference between DEVICE and HOST: %f\n", maxDiff);

	//Free host data
	free(a_host);
	free(b_host);
	free(c_host);
	free(res_host);
	return 0;
}