#ifndef __UTILS_H__
#define __UTILS_H__

void InitData(float *a, float *b, float *c, const size_t elementNum);
float CheckResults(const float *res_host, const float *res_dev, const size_t elementNum);
static bool AbsCompare(int a, int b);
void InitMatrixA(float *a, const size_t elementNum);
void InitMatrixB(float *b, const size_t elementNum);
void InitMatrixC(float *c, const size_t elementNum);
void RandomInit(float *data, int size);

#endif //__UTILS_H__
