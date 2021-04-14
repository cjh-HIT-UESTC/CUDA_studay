#include<stdio.h>
#include<stdlib.h>
#include<iostream>

//cuda的一些头文件
#include <cuda.h>
#include <curand_kernel.h>
#include<cuda_runtime_api.h>
#include "device_launch_parameters.h"

void cjh();  //调起函数   在cu文件里 调起kernal文件   要在主函数前声明

int main()
{
	cjh();//调起函数
}
