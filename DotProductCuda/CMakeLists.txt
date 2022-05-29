cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(DotProduct LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(WIN32)
	set(CUDA_SAMPLES_INCLUDE "C:\\ProgramData\\NVIDIA Corporation\\CUDA Samples\\v11.4\\common\\inc\\")
else()
	set(CUDA_SAMPLES_INCLUDE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/samples/common/inc/")
endif()

add_subdirectory(DotProductCuda)
add_subdirectory(DotProductCudaDynamicShared)
add_subdirectory(DotProductCudaOccupancy)
