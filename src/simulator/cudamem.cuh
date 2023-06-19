#pragma once
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <nvml.h>
// #include "../macro.h"
//CUDA错误检测函数
inline void CheckCall(cudaError_t err, const char *file, int line)
{
    const cudaError_t error = err;
    if (error != cudaSuccess)
    {
        printf("Error:%s.Line %d,", file, line);
        printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}
#define CUDACHECK(x) CheckCall(x, __FILE__, __LINE__)
//CUDA NVML错误检测函数
inline void CheckNVML(nvmlReturn_t result, const char *file, int line){
    if(result!=NVML_SUCCESS){
        printf("NVML Error:%s,\n",nvmlErrorString(result));
        exit(1);
    }
}
#define NVMLCHECK(x) CheckNVML(x, __FILE__, __LINE__)
//拷贝到GPU中
template<typename T>
T* toGPU(T* cpu, int size)
{
    T * ret;
    CUDACHECK(cudaMalloc((void**)&(ret), sizeof(T) * size));
    CUDACHECK(cudaMemcpy(ret, cpu, sizeof(T)*size, cudaMemcpyHostToDevice));
    return ret;
}


template<typename T>
void toGPU(T* cpu,T* gpu, int size)
{
    CUDACHECK(cudaMalloc((void**)&(gpu), sizeof(T) * size));
    CUDACHECK(cudaMemcpy(gpu, cpu, sizeof(T)*size, cudaMemcpyHostToDevice));
}
//拷贝回CPU
template<typename T>
T* toCPU(T* gpu, int size)
{
    T * ret = new T[size];
	CUDACHECK(cudaMemcpy(ret, gpu, sizeof(T)*size, cudaMemcpyDeviceToHost));
	return ret;
}

template<typename T>
void toCPU(T* gpu,T* cpu,int size)
{
    CUDACHECK(cudaMemcpy(cpu, gpu, sizeof(T)*size, cudaMemcpyDeviceToHost));
}
//释放GPU中内存空间
template<typename T>
void gpuFree(T* gpu)
{
	CUDACHECK(cudaFree(gpu));
}