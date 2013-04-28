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
#include <cuda.h>

#include "poly.h"

#define THREADS 1024


void cuda_init( int devID )
{
	cudaDeviceProp deviceProp;

	cudaGetDeviceProperties(&deviceProp, devID);
	if( deviceProp.major < 1 )
	{
		fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
		exit(-1);
	}
	cudaSetDevice(devID);
	fprintf(stderr, "gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);
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
		data_d[pos] = data_a[pos] * data_b[pos];
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
	ul_int i,num,tmp2;
	ul_int tmp = 0;
	ul_int pos = 0;
	ul_int sum = (blockIdx.x)*blockDim.x + (threadIdx.x) + 1;
	
	if(sum >= 2*(size-1)) return;

	num = (sum+1) >> 1;
	if( sum >= size ) pos = sum - size + 1;

	tmp2 = sum-pos;
	for( i=pos ; i<num ; i++ )
	{
		tmp += (data_a[i]+data_a[tmp2]) * (data_b[i]+data_b[tmp2]);
		tmp -= data_d[i];
		tmp -= data_d[tmp2--];
	}

	atomicAdd( &data_c[sum], tmp );
}

__global__ void C_zero( 
		ul_int * data_c,
		ul_int size )
{
	ul_int sum = (blockIdx.x)*blockDim.x + (threadIdx.x);
	
	if(sum >= 2*size) return;

	data_c[sum] = 0;
}

__global__ void calculate_misc( 
		ul_int * data_c,
		ul_int * data_d,
		ul_int size )
{
	data_c[threadIdx.x*2*(size-1)] = data_d[threadIdx.x*(size-1)];
}

__global__ void calculate_even( 
		ul_int * data_c,
		ul_int * data_d,
		ul_int size )
{
	ul_int sum = (blockIdx.x)*blockDim.x + (threadIdx.x) + 1;
	if(sum >= (size-1)) return;

	atomicAdd( &data_c[2*sum],  data_d[sum]);
}

extern "C" void calculate_cuda( void )
{
	ul_int size = poly_size[A];
	ul_int * data_a;
	ul_int * data_b;
	ul_int * data_c;
	ul_int * data_d;
	ul_int blocks = size/THREADS;
	cudaStream_t stream[2];
	cudaStreamCreate(&stream[0]);
	cudaStreamCreate(&stream[1]);

	if( size%THREADS != 0 ) blocks++;

	cuda_init(0);

	cudaMalloc( &data_b, sizeof(ul_int)*size );
	cudaMalloc( &data_a, sizeof(ul_int)*size );
	cudaMalloc( &data_c, 2*sizeof(ul_int)*size );
	cudaMalloc( &data_d, sizeof(ul_int)*size );
	cudaMemcpyAsync( data_a, poly[A], sizeof(ul_int)*size, cudaMemcpyHostToDevice, stream[1]);
	cudaMemcpyAsync( data_b, poly[B], sizeof(ul_int)*size, cudaMemcpyHostToDevice, stream[1]);
	C_zero<<<2*blocks,THREADS,0,stream[0]>>>( data_c, size );
	cudaStreamSynchronize( stream[0] );
	cudaStreamSynchronize( stream[1] );


	calculate_D<<<blocks,THREADS>>>( data_a, data_b, data_d, size );

	calculate_C<<<2*blocks,THREADS>>>( data_a, data_b, data_c, data_d, size );
	calculate_even<<<blocks,THREADS>>>( data_c, data_d, size );
  calculate_misc<<<1,2>>>( data_c, data_d, size);

	cudaMemcpy( poly[C], data_c, 2*sizeof(ul_int)*size, cudaMemcpyDeviceToHost );
 
	cudaStreamDestroy(stream[0]);
	cudaStreamDestroy(stream[1]);
	cudaFree( data_a );
	cudaFree( data_b );
	cudaFree( data_c );
}
