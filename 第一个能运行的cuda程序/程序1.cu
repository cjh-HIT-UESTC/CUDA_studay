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

void cjh()//调起函数
{
	//读出设备有多少个显卡
	int gpuCount = -1;			//显卡数量初始化
	cudaGetDeviceCount(&gpuCount);		//得到显卡数量 注意用了：&
	printf("gpuCount: %d\n", gpuCount);

	cudaSetDevice(gpuCount - 1);	//选择最后一块设备（本机上也就是第0号）

	//cudaSetDevice(0);						//选择设备
	float* aGpu;								//定义设备
	cudaMalloc((void**)&aGpu, 16 * sizeof(float));//分配显存
	float a[16] = { 0 };						//分配内存
	cudaMemcpy(aGpu, a, 16 * sizeof(float), cudaMemcpyHostToDevice);//主机到设备
	kernal << <1, 16 >> > (aGpu);								//调用线程函数
	cudaMemcpy(a, aGpu, 16 * sizeof(float), cudaMemcpyDeviceToHost);//设备到主机
	for (int i = 0; i < 16; ++i)		//输出结果
	{
		printf("%f\n", a[i]);
	}
	cudaFree(aGpu);		//释放显存
	cudaDeviceReset(); //把设定恢复初始化 便于后续换设备

	//查设备的一些信息
	cudaDeviceProp prop;	//定义一个结构体 这个结构体内部由一些数据构成
	cudaGetDeviceProperties(&prop, 0);
	printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim: %d\n", prop.maxThreadsDim);
	printf("maxGridSize: %d\n", prop.maxGridSize);
	printf("totalConstMem: %d\n", prop.totalConstMem);
	printf("clockRate: %d\n", prop.clockRate);
	printf("integrated: %d\n", prop.integrated);

	//自动选择最优的设备
	int deviceId;		//先定义设备编号
	cudaChooseDevice(&deviceId, &prop);	//传入定义的编号 和 查到的设备信息结构体 自动返回选到的设备编号到定义的数字里去
	printf("deviceId: %d\n", deviceId);

	//因为本机只有一个设备 如果要分配两个设备会报错 捕捉它的错误 并给出如何分配两个设备
	//int deviceList[2] = { 0,1 };
	//cudaSetValidDevices(deviceList, 1);




	
}



