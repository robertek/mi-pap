cmake_minimum_required( VERSION 2.8 )
project( POLY )
set( CMAKE_SUPPRESS_REGENERATION on )
set( CMAKE_VERBOSE_MAKEFILE no )
include_directories( "${PROJECT_BINARY_DIR}" )

# cuda options
set( CUDA_TOOLKIT_ROOT_DIR "/opt/cuda" )
find_package( CUDA )
set( CUDA_NVCC_FLAGS -arch=sm_20 )
set( OCELOT 1 ) # set 0 if ocelot not used
if( OCELOT )
	set( CUDA_LIBRARIES -locelot )
endif()
#

#set( CMAKE_C_FLAGS "-march=native -O3 -pipe -floop-block -floop-strip-mine -floop-interchange -flto" )
set( CMAKE_C_FLAGS "-O0 -ggdb -Wall -Wno-format-extra-args -fopenmp" )
link_libraries( -lm )



# fft
add_executable( poly_serial_fft src/poly.c src/fft.c )
SET_TARGET_PROPERTIES( poly_serial_fft
	PROPERTIES COMPILE_FLAGS "-DSERIAL" )

add_executable( poly_openmp_fft src/poly.c src/fft.c )
SET_TARGET_PROPERTIES( poly_openmp_fft 
	PROPERTIES COMPILE_FLAGS "-DOPENMP" )

if( CUDA_FOUND )
	cuda_add_executable( poly_cuda_fft src/poly.c src/fft.cu )
	SET_TARGET_PROPERTIES( poly_cuda_fft
		PROPERTIES COMPILE_FLAGS "-DCUDA" )
endif()
# end fft

# karatsuba
add_executable( poly_serial_karatsuba src/poly.c src/karatsuba.c )
SET_TARGET_PROPERTIES( poly_serial_karatsuba
	PROPERTIES COMPILE_FLAGS "-DSERIAL" )

add_executable( poly_openmp_karatsuba src/poly.c src/karatsuba.c )
SET_TARGET_PROPERTIES( poly_openmp_karatsuba 
	PROPERTIES COMPILE_FLAGS "-DOPENMP" )

if( CUDA_FOUND )
	cuda_add_executable( poly_cuda_karatsuba src/poly.c src/karatsuba.cu )
	SET_TARGET_PROPERTIES( poly_cuda_karatsuba 
		PROPERTIES COMPILE_FLAGS "-DCUDA" )
endif()
# end karatsuba

# naive
add_executable( poly_serial_naive src/poly.c src/naive.c )
SET_TARGET_PROPERTIES( poly_serial_naive
	PROPERTIES COMPILE_FLAGS "-DSERIAL" )

add_executable( poly_openmp_naive src/poly.c src/naive.c )
SET_TARGET_PROPERTIES( poly_openmp_naive 
	PROPERTIES COMPILE_FLAGS "-DOPENMP" )

if( CUDA_FOUND )
	cuda_add_executable( poly_cuda_naive src/poly.c src/naive.cu )
	SET_TARGET_PROPERTIES( poly_cuda_naive 
		PROPERTIES COMPILE_FLAGS "-DCUDA" )
endif()
#end naive



# tests
enable_testing()
