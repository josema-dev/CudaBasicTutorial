cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(TARGET "AddCuda")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(${TARGET} LANGUAGES CUDA)

add_executable(${TARGET} main.cu)

set_property(TARGET ${TARGET}
			 PROPERTY CUDA_SEPARABLE_COMPILATION ON)

if(APPLE)
	# We need to add the path to the driver (libcuda.dylib) as an rpath, 
	# so that the static cuda runtime can find it at runtime.
	set_property(TARGET ${TARGET} 
				 PROPERTY
				 BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()