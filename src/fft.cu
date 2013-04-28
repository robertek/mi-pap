/*
 * =====================================================================================
 *
 *       Filename: fft.cu
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
#include <cufft.h>

#include "poly.h"

#define C_SIZE (2*poly_size[0])
#define C_SIZE_COMPLEX (C_SIZE*sizeof(cufftComplex))

#define THREADS 1024

/* init and print GPU */
void cuda_init( int devID )
{
	cudaDeviceProp deviceProp;

	cudaGetDeviceProperties( &deviceProp, devID );
	if( deviceProp.major < 1 )
	{
		fprintf( stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n" );
		exit( -1 );
	}
	cudaSetDevice(devID);
	fprintf( stderr, "gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name );
}

/* just multiply complex numbers */
__global__ void multiply_points( cufftComplex * dev_a, cufftComplex * dev_b, cufftComplex * dev_c, ul_int size )
{
	ul_int i = (blockIdx.x)*blockDim.x + (threadIdx.x);

	if(i<size)
	{
		dev_c[i].x = dev_a[i].x * dev_b[i].x - dev_a[i].y * dev_b[i].y;
		dev_c[i].y = dev_a[i].x * dev_b[i].y + dev_a[i].y * dev_b[i].x;
	}
}

/* retype and copy poly to complex host_x and copy to GPU */
void copy_mem_dev( cufftComplex * host_a, cufftComplex * host_b, cufftComplex * dev_a, cufftComplex * dev_b )
{
	ul_int i;
	for( i=0 ; i<poly_size[0] ; i++ )
	{
		host_a[i].x = (float)poly[A][i];
		host_b[i].x = (float)poly[B][i];
	}

	cudaMemcpy( dev_a, host_a, C_SIZE_COMPLEX, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_b, host_b, C_SIZE_COMPLEX, cudaMemcpyHostToDevice );
}

/* copy array C from GPU and retype real part to int */
void copy_mem_host( cufftComplex * host_c, cufftComplex * dev_c )
{
	cudaMemcpy( host_c, dev_c, C_SIZE_COMPLEX, cudaMemcpyDeviceToHost );

	ul_int i;
	for( i=0 ; i<C_SIZE ; i++ )
	{
		poly[C][i] = (unsigned int)(host_c[i].x/C_SIZE);
	}
}

extern "C" void calculate_cuda( void )
{
	cufftHandle plan;
	cufftComplex * cu_poly_a;
	cufftComplex * cu_poly_b;
	cufftComplex * cu_poly_c;
	cufftComplex * cu_host_poly_a;
	cufftComplex * cu_host_poly_b;
	cufftComplex * cu_host_poly_c;
	ul_int blocks = C_SIZE/THREADS + 1;

	cuda_init( 0 );

	/* alloc required arrays */
	cudaMalloc( (void**)&cu_poly_a, C_SIZE_COMPLEX );
	cudaMalloc( (void**)&cu_poly_b, C_SIZE_COMPLEX );
	cudaMalloc( (void**)&cu_poly_c, C_SIZE_COMPLEX );
	cu_host_poly_a = (cufftComplex*) calloc( sizeof(cufftComplex), C_SIZE );
	cu_host_poly_b = (cufftComplex*) calloc( sizeof(cufftComplex), C_SIZE );
	cu_host_poly_c = (cufftComplex*) calloc( sizeof(cufftComplex), C_SIZE );

	/* copy data to GPU */
	copy_mem_dev( cu_host_poly_a, cu_host_poly_b, cu_poly_a, cu_poly_b );

	/* set up cufft */
	cufftPlan1d( &plan, C_SIZE, CUFFT_C2C, 1 );

	/* call fft transformation on each input array */
	cufftExecC2C( plan, cu_poly_a, cu_poly_a, CUFFT_FORWARD );
	cufftExecC2C( plan, cu_poly_b, cu_poly_b, CUFFT_FORWARD );

	/* multiply complex products of fft transformation */
	multiply_points<<<blocks,THREADS>>>( cu_poly_a, cu_poly_b, cu_poly_c, C_SIZE );

	/* call inverse fft transformation on output array */
	cufftExecC2C( plan, cu_poly_c, cu_poly_c, CUFFT_INVERSE );

	/* copy data from GPU */
	copy_mem_host( cu_host_poly_c, cu_poly_c );

	/* free memory */
	cufftDestroy( plan );
	cudaFree( cu_poly_a );
	cudaFree( cu_poly_b );
	cudaFree( cu_poly_c );
	free( cu_host_poly_a );
	free( cu_host_poly_b );
	free( cu_host_poly_c );
}
