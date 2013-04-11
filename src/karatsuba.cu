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

#include "poly.h"

/*
 * Di coefficient
 * Di = Ai * Bi
 */
__global__ void calculate_D(
		unsigned long int * data_a,
		unsigned long int * data_b,
		unsigned long int * data_d,
		unsigned long int size )
{
	unsigned long int i;
	for ( i=0 ; i<size ; i++ ) 
	{
		data_d[i] = data_a[i]*data_b[i];
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
		unsigned long int * data_a,
		unsigned long int * data_b,
		unsigned long int * data_c,
		unsigned long int * data_d,
		unsigned long int size )
{
	unsigned long int i,pos,num,sum;
	for ( sum=1 ; sum<2*(size-1) ; sum++ ) 
	{
		num = (sum+1) >> 1;
		data_c[sum] = 0;

		if( sum < size ) pos=0;
		else pos = sum - size + 1;

		for( i=pos ; i<num ; i++ )
		{
			data_c[sum] += (data_a[i] + data_a[sum-i]) * (data_b[i] + data_b[sum-i]);
			data_c[sum] -= data_d[i];
			data_c[sum] -= data_d[sum-i];
		}

		if( sum%2 == 0 )
		{
			data_c[sum] += data_d[sum/2];
		}
	}

	/* set first and last number */
	data_c[0]=data_d[0];
	data_c[2*(size-1)]=data_d[size-1];
}

extern "C" void calculate_cuda( void )
{
	unsigned long int size = poly_size[A];
	unsigned long int * data_a;
	unsigned long int * data_b;
	unsigned long int * data_c;
	unsigned long int * data_d;

	cudaMalloc( &data_a, sizeof(long int)*size );
	cudaMalloc( &data_b, sizeof(long int)*size );
	cudaMalloc( &data_d, sizeof(long int)*size );
	cudaMalloc( &data_c, 2*sizeof(long int)*size );
	cudaMemcpy( data_a, poly[A], sizeof(long int)*size, cudaMemcpyHostToDevice );
	cudaMemcpy( data_b, poly[B], sizeof(long int)*size, cudaMemcpyHostToDevice );

	calculate_D<<<1,1>>>( data_a, data_b, data_d, size );
	calculate_C<<<1,1>>>( data_a, data_b, data_c, data_d, size );

	cudaMemcpy( poly[C], data_c, 2*sizeof(long int)*size, cudaMemcpyDeviceToHost );
	cudaFree( data_a );
	cudaFree( data_b );
	cudaFree( data_c );
}
