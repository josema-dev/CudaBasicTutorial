#include <cstdio>

#include <cuda_runtime.h>

__device__ float DeviceFunction(int thIdx)
{
	printf("Hi from device function!\n");
	return 0.1f * (thIdx + 1);
}

//Kernel definition
__global__ void HelloWorld()
{
	printf("Hello World from thread %d\nNumber from device: %f\n", threadIdx.x, DeviceFunction(threadIdx.x));
}

int main()
{
	constexpr int N = 2;
	
	//Kernel invocation with N threads.
	HelloWorld<<<1,N>>>();
	cudaDeviceSynchronize();
	return 0;
}