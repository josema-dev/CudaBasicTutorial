cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(HelloWorld LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_subdirectory(HelloWorldGcc)
add_subdirectory(HelloWorldCuda)
add_subdirectory(HelloWorldCudaLinuxFix)