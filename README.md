# CUDA_studay
记录学习CUDA编程的过程

如何创建一个新的cuda工程：
  1、新建空项目
  2、右键工程名，生成依赖项
  3、创建。cu文件
  4、包含头文件：
          #include<stdio.h>
          #include<stdlib.h>
          #include<iostream>

          //cuda的一些头文件
          #include <cuda.h>
          #include <curand_kernel.h>
          #include<cuda_runtime_api.h>
          #include "device_launch_parameters.h"
