cmake_minimum_required( VERSION 2.8 )
project( POLY )
set( CMAKE_SUPPRESS_REGENERATION on )
set( CMAKE_VERBOSE_MAKEFILE no )
include_directories( "${PROJECT_BINARY_DIR}" )

#set( CMAKE_C_FLAGS "-march=native -O3 -pipe -floop-block -floop-strip-mine -floop-interchange -flto" )
set( CMAKE_C_FLAGS "-O0 -ggdb -Wall -Wno-format-extra-args" )

add_executable( poly src/poly.c src/naive.c )

# mpi
#find_package(MPI REQUIRED)
#SET(CMAKE_C_COMPILER mpicc)
#set(CMAKE_C_COMPILE_FLAGS ${CMAKE_C_FLAGS} ${MPI_C_FLAGS})
#set(CMAKE_C_LINK_FLAGS ${CMAKE_C_LINK_FLAGS} ${MPI_LINK_FLAGS})
#include_directories(MPI_INCLUDE_PATH)
#target_link_libraries(sop ${MPI_LIBRARIES})

# tests
enable_testing()
