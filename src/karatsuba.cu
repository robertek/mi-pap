/*
 * =====================================================================================
 *
 *       Filename:  karatsuba.cu
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
#include <stdio.h>

#include <cuda_runtime.h>

#include "poly.h"

#define THREADS 512

void cuda_init( void )
{
  int devID=0;
	cudaDeviceProp deviceProp;

	cudaGetDeviceProperties(&deviceProp, devID);
	if( deviceProp.major < 1 )
	{
		fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
		exit(-1);
	}
	cudaSetDevice(devID);
	printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);
}

/*
 * Di coefficient
 * Di = Ai * Bi
 */
__global__ void calculate_D(
		ul_int * data_a,
		ul_int * data_b,
		ul_int * data_d,
		ul_int size )
{
	ul_int pos = (blockIdx.x)*blockDim.x + (threadIdx.x);
	
	if( pos<size )
	{
		data_d[pos] = data_a[pos]*data_b[pos];
	}
}

/*
 * Ci coefficient
 *
 * Ci = sum( Dpq ) - sum(Dp + Dq)
 * Dpq = (Ap+Aq)*(Bp+Bq)
 *
 * where p+q = i and p>q>=0
 * 
 * ex: 1 = 0+1        # poly_size = 5
 *     2 = 0+2
 *     3 = 0+3 | 1+2
 *     4 = 0+4 | 1+3
 *     5 = 1+4 | 2+3
 *     6 = 2+4
 *     7 = 3+4
 *
 * for even i: Ci += D[i/2]
 *
 * sum = alias for "i"
 */
__global__ void calculate_C( 
		ul_int * data_a,
		ul_int * data_b,
		ul_int * data_c,
		ul_int * data_d,
		ul_int size )
{
	ul_int i,pos,num,sum;

	sum = (blockIdx.x)*blockDim.x + (threadIdx.x);

	data_c[sum] = 0;

	if(sum == 0)
	{
		data_c[0]=data_d[0];
		return;
	}
	if(sum == 2*(size-1))
	{
		data_c[2*(size-1)]=data_d[size-1];
		return;
	}

	if( sum%2 == 0 )
	{
		data_c[sum] += data_d[sum/2];
	}

	num = (sum+1) >> 1;

	if( sum < size ) pos=0;
	else pos = sum - size + 1;

	for( i=pos ; i<num ; i++ )
	{
		data_c[sum] += (data_a[i] + data_a[sum-i]) * (data_b[i] + data_b[sum-i]);
		data_c[sum] -= data_d[i];
		data_c[sum] -= data_d[sum-i];
	}
}

extern "C" void calculate_cuda( void )
{
	ul_int size = poly_size[A];
	ul_int * data_a;
	ul_int * data_b;
	ul_int * data_c;
	ul_int * data_d;
	ul_int blocks = size/THREADS;

	if( size%THREADS != 0 ) blocks++;

	cuda_init();

	cudaMalloc( &data_a, sizeof(ul_int)*size );
	cudaMemcpy( data_a, poly[A], sizeof(ul_int)*size, cudaMemcpyHostToDevice );
	cudaMalloc( &data_b, sizeof(ul_int)*size );
	cudaMemcpy( data_b, poly[B], sizeof(ul_int)*size, cudaMemcpyHostToDevice );
	cudaMalloc( &data_d, sizeof(ul_int)*size );
	cudaMalloc( &data_c, 2*sizeof(ul_int)*size );

	calculate_D<<<blocks,THREADS>>>( data_a, data_b, data_d, size );

	calculate_C<<<blocks,THREADS>>>( data_a, data_b, data_c, data_d, size );

	cudaMemcpy( poly[C], data_c, 2*sizeof(ul_int)*size, cudaMemcpyDeviceToHost );
	cudaFree( data_a );
	cudaFree( data_b );
	cudaFree( data_c );
}
