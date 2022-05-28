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
	memset((void*)c, 0, sizeof(float) * elementNum);
	for(size_t i=0; i<elementNum; i++)
	{
		a[i] = rand() / (float)RAND_MAX;
		b[i] = rand() / (float)RAND_MAX;
	}
}