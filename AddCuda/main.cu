#include <cstdio>

#include <cuda_runtime.h>

//kernel definition
__global__ void Add(float a, float b, float *c)
{
	*c = a + b;
}

int main()
{
	float *c_host, *c_dev;
	
	float a = 1.2;
	float b = 3.2;

	//allocate data
	c_host = (float*)malloc(sizeof(float));
	cudaMalloc(&c_dev, sizeof(float));
	
	//kernel lunch
	Add<<<1,1>>>(a, b, c_dev);
	
	cudaMemcpy(c_host, c_dev, sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("%f + %f = %f\n", a, b, *c_host);

	//free data
	cudaFree(c_dev);
	free(c_host);
	return 0;
}