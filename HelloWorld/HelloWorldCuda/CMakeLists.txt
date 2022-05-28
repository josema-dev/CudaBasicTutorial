set(TARGET "HelloWorldCuda")

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