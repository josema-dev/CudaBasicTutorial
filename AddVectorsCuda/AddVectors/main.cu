#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "utils.h"

//Kernel function
__global__ void AddVectors(float *a, float *b, float *c)
{
	int idx = threadIdx.x;
	c[idx] = a[idx] + b[idx];
}

//Host Vector Add
void AddVectorsHost(float *a, float *b, float *c, const size_t elementNum)
{
	for(size_t i=0; i<elementNum; i++)
	{
		c[i] = a[i] + b[i];
	}
}

int main()
{
	//Vector size
	constexpr size_t VECTOR_SIZE = 256;

	float *a_host, *b_host, *c_host, *res_host;
	float *a_dev, *b_dev, *c_dev;

	//Alloc host data
	a_host = (float*)malloc(sizeof(float) * VECTOR_SIZE);
	b_host = (float*)malloc(sizeof(float) * VECTOR_SIZE);
	c_host = (float*)malloc(sizeof(float) * VECTOR_SIZE);
	res_host = (float*)malloc(sizeof(float) * VECTOR_SIZE);

	//Alloc device data
	checkCudaErrors(cudaMalloc(&a_dev, sizeof(float) * VECTOR_SIZE));
	checkCudaErrors(cudaMalloc(&b_dev, sizeof(float) * VECTOR_SIZE));
	checkCudaErrors(cudaMalloc(&c_dev, sizeof(float) * VECTOR_SIZE));

	//Init host data
	InitData(a_host, b_host, c_host, VECTOR_SIZE);

	//Copy host data to dev
	checkCudaErrors(cudaMemcpy(a_dev, a_host, sizeof(float) * VECTOR_SIZE, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(b_dev, b_host, sizeof(float) * VECTOR_SIZE, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(c_dev, 0, sizeof(float) * VECTOR_SIZE));

	//Compute device
	AddVectors<<<1, VECTOR_SIZE>>>(a_dev, b_dev, c_dev);

	cudaError_t errorNum = cudaGetLastError();
	if(errorNum != cudaSuccess)
		printf("Cuda error running kernel.\n%s\n", cudaGetErrorName(errorNum));
	
	//Copy data back to host
	checkCudaErrors(cudaMemcpy(res_host, c_dev, sizeof(float) * VECTOR_SIZE, cudaMemcpyDeviceToHost));

	//Compute host
	AddVectorsHost(a_host, b_host, c_host, VECTOR_SIZE);

	//Check results
	float maxDiff = CheckResults(c_host, res_host, VECTOR_SIZE);

	printf("Max difference between DEVICE and HOST: %f\n", maxDiff);

	//Free device data
	checkCudaErrors(cudaFree(a_dev));
	checkCudaErrors(cudaFree(b_dev));
	checkCudaErrors(cudaFree(c_dev));

	//Free host data
	free(a_host);
	free(b_host);
	free(c_host);
	free(res_host);
	return 0;
}