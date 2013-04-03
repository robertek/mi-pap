/*
 * =====================================================================================
 *
 *       Filename:  naive.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  13.2.2013 22:59:33
 *       Compiler:  gcc
 *
 *         Author:  Robert David
 *
 * =====================================================================================
 */
#include <stdlib.h>

#include "poly.h"

__global__ void multiply( unsigned long int * data_a, 
		unsigned long int * data_b, unsigned long int * data_c, unsigned long int size )
{
	int i,j;
	unsigned long int tmp;

	i = threadIdx.x;
	j = blockIdx.x;

	data_c[i+j] = data_a[i]*data_b[j];
	//tmp = data_a[i]*data_b[j];
	//atomicAdd( &data_c[i+j], tmp );
}

extern "C" void calculate_cuda( void )
{
	unsigned long int * data_a;
	unsigned long int * data_b;
	unsigned long int * data_c;
	cudaMalloc( &data_a, sizeof(long int)*poly_size[A] );
	cudaMalloc( &data_b, sizeof(long int)*poly_size[A] );
	cudaMalloc( &data_c, 2*sizeof(long int)*poly_size[A] );

	cudaMemcpy( data_a, poly[A], sizeof(long int)*poly_size[A], cudaMemcpyHostToDevice );
	cudaMemcpy( data_b, poly[B], sizeof(long int)*poly_size[A], cudaMemcpyHostToDevice );

	multiply<<<poly_size[A],poly_size[A]>>>( data_a, data_b, data_c, poly_size[A] );

	cudaMemcpy( poly[C], data_c, 2*sizeof(long int)*poly_size[A], cudaMemcpyDeviceToHost );

	cudaFree( data_a );
	cudaFree( data_b );
	cudaFree( data_c );
}
