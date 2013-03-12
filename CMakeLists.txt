cmake_minimum_required( VERSION 2.8 )
project( POLY )
set( CMAKE_SUPPRESS_REGENERATION on )
set( CMAKE_VERBOSE_MAKEFILE no )
include_directories( "${PROJECT_BINARY_DIR}" )

# cuda options
set( CUDA_TOOLKIT_ROOT_DIR "/opt/cuda" )
find_package( CUDA )
set( OCELOT 1 ) # set 0 if ocelot not used
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
	if( OCELOT )
		#target_link_libraries( poly_cuda_fft -locelot )
	endif()
	cuda_add_executable( poly_cuda_fft src/poly.c src/fft.cu )
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
	if( OCELOT )
		#target_link_libraries( poly_cuda_karatsuba -locelot )
	endif()
	cuda_add_executable( poly_cuda_karatsuba src/poly.c src/karatsuba.cu )
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
	if( OCELOT )
		#target_link_libraries( poly_cuda_naive -locelot )
	endif()
	cuda_add_executable( poly_cuda_naive src/poly.c src/naive.cu )
endif()
#end naive




# tests
enable_testing()
