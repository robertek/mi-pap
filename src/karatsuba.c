/*
 * =====================================================================================
 *
 *       Filename:  karatsuba.c
 *
 *    Description:  implement multiply with karatsuba algorighm
 *
 *    							good info at:
 *    							http://www.ijcse.com/docs/INDJCSE12-03-01-082.pdf
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

/* array for helper coefficients */
unsigned long int * D; 

void alloc_arrays( void )
{
	D = (unsigned long int *) calloc( sizeof(long int), poly_size[0] );
}

void free_arrays( void )
{
	free( D );
}

/*
 * Di coefficient
 * Di = Ai * Bi
 */
inline void calculate_D( unsigned long int pos )
{
	D[pos] = poly[A][pos]*poly[B][pos];
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
void calculate_C( unsigned long int sum )
{
	unsigned long int num = (sum+1) >> 1;
	unsigned long int i,pos;

	if( sum - poly_size[A] < 0 )
	{
		pos=0;
	}
	else
	{
		pos= sum - poly_size[A] + 1;
	}

	for( i=pos ; i<num ; i++ )
	{
		poly[C][sum] += (poly[A][i] + poly[A][sum-i]) * (poly[B][i] + poly[B][sum-i]);
		poly[C][sum] -= D[i];
		poly[C][sum] -= D[sum-i];
	}

	if( sum%2 == 0 )
	{
		poly[C][sum] += D[sum/2];
	}
}

#if defined SERIAL
void calculate_serial( void )
{
	unsigned long int i;

	alloc_arrays();

	for ( i=0 ; i<poly_size[0] ; i++ ) 
	{
		calculate_D( i );
	}

	for ( i=1 ; i<2*(poly_size[0]-1) ; i++ ) 
	{
		calculate_C( i );
	}

	/* set first and last number */
	poly[C][0]=D[0];
	poly[C][2*(poly_size[0]-1)]=D[poly_size[0]-1];

	free_arrays();
}
#endif

#if defined OPENMP
void calculate_openmp( void )
{
	unsigned long int i;
	unsigned long int size = poly_size[A];

	alloc_arrays();

#pragma omp parallel for private(i)
	for ( i=0 ; i<size ; i++ ) 
	{
		calculate_D( i );
	}

	for ( i=1 ; i<2*(size-1) ; i++ ) 
	{
		calculate_C( i );
	}

	/* set first and last number */
	poly[C][0]=D[0];
	poly[C][2*(size-1)]=D[size-1];

	free_arrays();
}
#endif

