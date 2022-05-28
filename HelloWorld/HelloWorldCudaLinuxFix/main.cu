#include <cstdio>

#include <cuda_runtime.h>

//Kernel definition
__global__ void  HelloWorld()
{
	printf("Hello World from thread %d\n", threadIdx.x);
}

int main()
{
	constexpr int N = 2;
	
	//Kernel invocation with N threads.
	HelloWorld<<<1,N>>>();
	cudaDeviceSynchronize();
	return 0;
}