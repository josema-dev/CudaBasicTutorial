#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "utils.h"

__global__ void AddVectors(float *a, float *b, float *c, const size_t elementNum)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx<elementNum)
		c[idx] = a[idx] + b[idx];
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
	constexpr size_t STREAMS_NUM = 2;
	constexpr size_t VECTOR_SIZE = 40960;
	constexpr size_t FULL_SIZE = VECTOR_SIZE * STREAMS_NUM;
	constexpr size_t THREADS_PER_BLOCK = 256;
	float *a_host, *b_host, *c_host, *res_host;
	float *a_dev[STREAMS_NUM], *b_dev[STREAMS_NUM], *c_dev[STREAMS_NUM];

	cudaEvent_t start, stop;
	cudaStream_t streams[STREAMS_NUM];

	// Alloc host data
	//a_host = (float*)malloc(sizeof(float) * VECTOR_SIZE * STREAMS_NUM);
	checkCudaErrors(cudaMallocHost(&a_host, sizeof(float) * VECTOR_SIZE * STREAMS_NUM));
	//b_host = (float *)malloc(sizeof(float) * VECTOR_SIZE * STREAMS_NUM);
	checkCudaErrors(cudaMallocHost(&b_host, sizeof(float) * VECTOR_SIZE * STREAMS_NUM));
	//c_host = (float *)malloc(sizeof(float) * VECTOR_SIZE * STREAMS_NUM);
	checkCudaErrors(cudaMallocHost(&c_host, sizeof(float) * VECTOR_SIZE * STREAMS_NUM));
	//res_host = (float *)malloc(sizeof(float) * VECTOR_SIZE * STREAMS_NUM);
	checkCudaErrors(cudaMallocHost(&res_host, sizeof(float) * VECTOR_SIZE * STREAMS_NUM));

	// Init host data
	InitData(a_host, b_host, c_host, VECTOR_SIZE * STREAMS_NUM);

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	for (int i = 0; i < STREAMS_NUM; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i]));
	
	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < STREAMS_NUM; i++)
	{
		// Alloc device data
		checkCudaErrors(cudaMallocAsync(&a_dev[i], sizeof(float) * VECTOR_SIZE, streams[i]));
		checkCudaErrors(cudaMallocAsync(&b_dev[i], sizeof(float) * VECTOR_SIZE, streams[i]));
		checkCudaErrors(cudaMallocAsync(&c_dev[i], sizeof(float) * VECTOR_SIZE, streams[i]));

		//Copy host data to dev
		checkCudaErrors(cudaMemcpyAsync(a_dev[i], a_host + i, sizeof(float) * VECTOR_SIZE, cudaMemcpyHostToDevice, streams[i]));
		checkCudaErrors(cudaMemcpyAsync(b_dev[i], b_host + i, sizeof(float) * VECTOR_SIZE, cudaMemcpyHostToDevice, streams[i]));
		checkCudaErrors(cudaMemsetAsync(c_dev[i], 0, sizeof(float) * VECTOR_SIZE, streams[i]));

		//Compute device
		dim3 blocks(1,1,1);
		dim3 threads(1,1,1);

		threads.x = THREADS_PER_BLOCK;
		blocks.x = (VECTOR_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

		AddVectors<<<blocks, threads, 0, streams[i]>>>(a_dev[i], b_dev[i], c_dev[i], VECTOR_SIZE);

		cudaError_t errorNum = cudaGetLastError();
		if(errorNum != cudaSuccess)
			printf("Cuda error running kernel.\n%s\n", cudaGetErrorName(errorNum));
		
		//Copy data back to host
		checkCudaErrors(cudaMemcpyAsync(res_host+i, c_dev[i], sizeof(float) * VECTOR_SIZE, cudaMemcpyDeviceToHost, streams[i]));

		//Free device data
		checkCudaErrors(cudaFreeAsync(a_dev[i], streams[i]));
		checkCudaErrors(cudaFreeAsync(b_dev[i], streams[i]));
		checkCudaErrors(cudaFreeAsync(c_dev[i], streams[i]));
	}
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	float elapsedTime = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Cuda execution time: %f ms.\n", elapsedTime);

	for (int i = 0; i < STREAMS_NUM; i++)
		checkCudaErrors(cudaStreamDestroy(streams[i]));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	// Compute host
	AddVectorsHost(a_host, b_host, c_host, VECTOR_SIZE);

	//Check results
	float maxDiff = CheckResults(c_host, res_host, VECTOR_SIZE);

	printf("Max difference between DEVICE and HOST: %f\n", maxDiff);

	//Free host data
	//free(a_host);
	checkCudaErrors(cudaFreeHost(a_host));
	//free(b_host);
	checkCudaErrors(cudaFreeHost(b_host));
	//free(c_host);
	checkCudaErrors(cudaFreeHost(c_host));
	//free(res_host);
	checkCudaErrors(cudaFreeHost(res_host));
	return 0;
}