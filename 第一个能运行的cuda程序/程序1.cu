#include<stdio.h>
#include<stdlib.h>
#include<iostream>

//cuda的一些头文件
#include <cuda.h>
#include <curand_kernel.h>
#include<cuda_runtime_api.h>
#include "device_launch_parameters.h"

__global__ void kernal(float* a)  //线程函数 
{
	a[threadIdx.x] = 1;
}

void cjh()                         //调起函数
{
	cudaSetDevice(0);              //选择设备
	float* aGpu;				//定义设备
	cudaMalloc((void**)&aGpu, 16 * sizeof(float));//分配显存
	float a[16] = { 0 };                     //分配内存
	cudaMemcpy(aGpu, a, 16 * sizeof(float), cudaMemcpyHostToDevice);//主机到设备
	kernal << <1, 16 >> > (aGpu);								//调用线程函数
	cudaMemcpy(a, aGpu, 16 * sizeof(float), cudaMemcpyDeviceToHost);//设备到主机
	for (int i = 0; i < 16; ++i)		//输出结果
	{
		printf("%f", a[i]);
	}
}



