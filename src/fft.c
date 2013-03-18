/*
 * =====================================================================================
 *
 *       Filename:  fft.c
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

#include "poly.h"
#include "math.h"
#include "complex.h"


#define C_SIZE (2*poly_size[0])

/* complex values for polynome interpretation */
double complex ** value;
/* precalculated roots of unity */
double complex * omega;
/* bit size of max polynom degree */
int count_bit_size;

void alloc_arrays( void )
{
	value = (double complex**) malloc( 3*sizeof(double complex*) );
	/* alloc 1 more item because it cause core when free ?!? */
	value[0] = (double complex*) calloc( sizeof(double complex), C_SIZE+1 );
	value[1] = (double complex*) calloc( sizeof(double complex), C_SIZE+1 );
	value[2] = (double complex*) calloc( sizeof(double complex), C_SIZE+1 );

	omega = (double complex*) calloc( sizeof(double complex), C_SIZE+1 );
}

void free_arrays( void )
{
	free(value[0]);
	free(value[1]);
	free(value[2]);
	free(value);

	free(omega);
}

#ifdef DEBUG
void print_values( int array )
{
	int i=C_SIZE;

	DEBUG( "DEBUG: array=%d\n", array );

	while(i--)
	{
		DEBUG( "%g %gi\n", __real__ value[array][i], __imag__ value[array][i] );
		DEBUG( "%g %gi\n", __real__ omega[i], __imag__ omega[i] );
	}
}
#endif


/* middle step after fft, just multiply each point (convolution) */
void multiply_points( void )
{
	int i;
	for( i=0 ; i<C_SIZE ; i++ ) 
	{
		value[C][i] = value[A][i]*value[B][i];
	}
}

/* precalculate roots of unity */
void calculate_omega( void )
{
	int i;

	/*
	 * omega[i] ... w^i   
	 * w^0 = 1
	 * w^i = w^(i-1) * exp(2*pi*I/n)
	 */
	omega[0]=1;

	for( i=1 ; i<C_SIZE ; i++ )
	{
		/* atan(1.0) = pi/4 */
		omega[i]=omega[i-1] * cexp(8*atan(1.0)*I/C_SIZE);
	}
}

/* calculate inverse bits 0001 => 1000 */
int inverse_bits( unsigned int num )
{
	unsigned int r = num;
	unsigned int tmp = 0;
	int s = count_bit_size-1;

	for( num>>=1 ; num ; num>>=1 )
	{
		r <<= 1;
		r |= num & 1;
		s--;
	}
	r <<= s;

	/* fill with zeros upper than count_bit_size */
	s=1;
	s <<= count_bit_size;
	while( s>>=1 ) tmp |= r & s;

	return tmp;
}

/* copy poly coefficients to complex array in correct position (butterfly) */
void copy_with_bit_inverse( int array )
{
	unsigned int i,inverse;

	for( i=0 ; i<poly_size[A] ; i++ )
	{
		inverse = inverse_bits( i );
		__real__ value[array][inverse] = poly[array][i];
	}
}

void fft( int array )
{
	int block_size=1;
	double complex tmp_omega,tmp1,tmp2;
	int i,j;

	while( block_size<C_SIZE )
	{
		for( i=0 ; i<C_SIZE ; i+=2*block_size )
		{
			for( j=0 ; j<block_size ; j++ )
			{
				tmp_omega = omega[(C_SIZE*j)/(2*block_size)];
				tmp1 = value[array][i+j] + tmp_omega*value[array][i+j+block_size];
				tmp2 = value[array][i+j] - tmp_omega*value[array][i+j+block_size];
				value[array][i+j] = tmp1;
				value[array][i+j+block_size] = tmp2;
			}
		}
		block_size <<= 1;
	}
}

/* inverse fft is normal fft with some modification, eg: w^-1 */
void inverse_fft()
{
	unsigned int i;
	unsigned int inverse;
	double complex tmp;

	/* we multiply now with w^-1, which means reverse the array from 1th position */
	for( i=1 ; i<poly_size[0]-1 ; i++ )
	{
		tmp = omega[C_SIZE-i];
		omega[C_SIZE-i] = omega[i];
		omega[i] = tmp;
	}

	fft( C );

	/* now back to inegers in correct position */
	for( i=0 ; i<C_SIZE ; i++ )
	{
		inverse = inverse_bits( i );
		poly[C][inverse] = floor(__real__ value[C][i]/C_SIZE + 0.001);
	}
}

#if defined SERIAL
void calculate_serial( void )
{
	/* calculate bit size of maximal poly degree */
	int i=C_SIZE;
	while( i>>=1 ) count_bit_size++;

	alloc_arrays();

	calculate_omega();

	/* copy data to complex arrays on correct positions */
	copy_with_bit_inverse( 0 );
	copy_with_bit_inverse( 1 );

	fft( 0 );
#ifdef DEBUG
	print_values( 0 );
#endif
	fft( 1 );
#ifdef DEBUG
	print_values( 1 );
#endif
	multiply_points();
#ifdef DEBUG
	print_values( 2 );
#endif
	inverse_fft();
#ifdef DEBUG
	print_values( 2 );
#endif

	free_arrays();
}
#endif

#if defined OPENMP
void calculate_openmp( void )
{
}
#endif
