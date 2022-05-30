set(TARGET "AddVectorsGrid")

add_executable(${TARGET} main.cu ${CMAKE_BINARY_DIR}/../../Utils/utils.cpp)

set_property(TARGET ${TARGET}
			 PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${TARGET} PRIVATE ${CUDA_SAMPLES_INCLUDE} PRIVATE ${CMAKE_BINARY_DIR}/../../Utils)
target_compile_options(${TARGET} PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G>")

if(APPLE)
	# We need to add the path to the driver (libcuda.dylib) as an rpath, 
	# so that the static cuda runtime can find it at runtime.
	set_property(TARGET ${TARGET} 
				 PROPERTY
				 BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()