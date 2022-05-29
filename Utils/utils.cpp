#include <vector>
#include <cmath>
#include <algorithm>

static bool AbsCompare(int a, int b)
{
    return (std::abs(a) < std::abs(b));
}

float CheckResults(const float *res_host, const float *res_dev, const size_t elementNum)
{
	std::vector<float> diff(elementNum);
	for(int i=0; i<elementNum; i++)
	{
		diff[i] = res_host[i] - res_dev[i];
	}
	return *std::max_element(diff.begin(), diff.end(), AbsCompare);
}

void InitData(float *a, float *b, float *c, const size_t elementNum)
{
	if(c)
		memset((void*)c, 0, sizeof(float) * elementNum);
	for(size_t i=0; i<elementNum; i++)
	{
		if(a)
			a[i] = rand() / (float)RAND_MAX;
		if(b)
			b[i] = rand() / (float)RAND_MAX;
	}
}

void InitMatrixA(float *a, const size_t elementNum)
{
	for(size_t i=0; i<elementNum; i++)
	{
		a[i] = i + 1.0f;
	}
}

void InitMatrixB(float *b, const size_t elementNum)
{
	for(size_t i=0; i<elementNum; i++)
	{
		b[i] = i + 2.0f;
	}
}

void InitMatrixC(float *c, const size_t elementNum)
{
	memset((void*)c, 0, sizeof(float) * elementNum);
}

// Allocates a matrix with random float entries.
void RandomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}
