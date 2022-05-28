cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(StreamsCuda LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(TARGET "StreamsCuda")

add_executable(${TARGET} main.cu ${CMAKE_BINARY_DIR}/../../Utils/utils.cpp)

if(WIN32)
	set(CUDA_SAMPLES_INCLUDE "C:\\ProgramData\\NVIDIA Corporation\\CUDA Samples\\v11.4\\common\\inc\\")
else()
	set(CUDA_SAMPLES_INCLUDE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/samples/common/inc/")
endif()

target_include_directories(${TARGET} PRIVATE ${CUDA_SAMPLES_INCLUDE} PRIVATE ${CMAKE_BINARY_DIR}/../../Utils)
target_compile_options(${TARGET} PRIVATE "--ptxas-options=-v" PRIVATE "$<$<CONFIG:DEBUG>:-G>")

if(APPLE)
	# We need to add the path to the driver (libcuda.dylib) as an rpath, 
	# so that the static cuda runtime can find it at runtime.
	set_property(TARGET ${TARGET}
				 PROPERTY
				 BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()