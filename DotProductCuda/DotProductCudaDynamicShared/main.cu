#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "utils.h"

__global__ void Dot(float *a, float *b, float *c, size_t elementsNum)
{
	//shared memory
	extern __shared__ float blockData[];

	//local memory
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t cacheIdx = threadIdx.x;
	float tmp = 0.0f;

	if(idx < elementsNum)
	{
		tmp += a[idx] * b[idx];
	}
	blockData[cacheIdx] = tmp;
	
	//sync all threads in block
	__syncthreads();

	//threads per block must be a power of 2
	int i = blockDim.x / 2;
	while(i != 0)
	{
		if(cacheIdx < i)
		{
			blockData[cacheIdx] += blockData[cacheIdx + i];
		}
		__syncthreads();
		i /= 2;
	}

	if(cacheIdx == 0)
	{
		c[blockIdx.x] = blockData[0];
	}
}

float DotHost(float *a, float *b, size_t elementsNum)
{
	float res = 0.0f;
	for (size_t i = 0; i < elementsNum; i++)
	{
		res += (a[i] * b[i]);
	}

	return res;
}

int main()
{
	constexpr size_t THREADS_PER_BLOCK = 32;
	constexpr size_t VECTOR_SIZE = 256;

	float *a_host, *b_host, *res_host;
	float *a_dev, *b_dev, *c_dev;

	a_host = (float *)malloc(sizeof(float) * VECTOR_SIZE);
	b_host = (float *)malloc(sizeof(float) * VECTOR_SIZE);
	res_host = (float *)malloc(sizeof(float) * VECTOR_SIZE);

	checkCudaErrors(cudaMalloc(&a_dev, sizeof(float) * VECTOR_SIZE));
	checkCudaErrors(cudaMalloc(&b_dev, sizeof(float) * VECTOR_SIZE));

	InitData(a_host, b_host, nullptr, VECTOR_SIZE);

	checkCudaErrors(cudaMemcpy(a_dev, a_host, sizeof(float) * VECTOR_SIZE, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(b_dev, b_host, sizeof(float) * VECTOR_SIZE, cudaMemcpyHostToDevice));

	dim3 threads, blocks;
	threads.x = THREADS_PER_BLOCK;
	blocks.x = (VECTOR_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	
	checkCudaErrors(cudaMalloc(&c_dev, sizeof(float) * blocks.x));

	Dot<<<blocks, threads, THREADS_PER_BLOCK * sizeof(float)>>>(a_dev, b_dev, c_dev, VECTOR_SIZE);

	checkCudaErrors(cudaMemcpy(res_host, c_dev, sizeof(float) * blocks.x, cudaMemcpyDeviceToHost));

	float dot_host = DotHost(a_host, b_host, VECTOR_SIZE);

	float c = 0;
 	for (int i=0; i<blocks.x; i++) {
 		c += res_host[i];
 	}

	printf("CPU res: %f, GPU res: %f, diff: %f\n", dot_host, c, std::abs(c - dot_host));
	return 0;
}