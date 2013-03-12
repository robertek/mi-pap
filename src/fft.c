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

complex ** value;
complex * omega;
int count_bit_size;

void alloc_arrays( void )
{
	value = (complex**) malloc( 3*sizeof(complex*) );
	value[0] = (complex*) calloc( sizeof(complex), C_SIZE+1 );
	value[1] = (complex*) calloc( sizeof(complex), C_SIZE+1 );
	value[2] = (complex*) calloc( sizeof(complex), C_SIZE+1 );

	omega = (complex*) calloc( sizeof(complex), C_SIZE+1 );
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

void multiply_points( void )
{
	int i;
	for( i=0 ; i<C_SIZE ; i++ ) 
	{
		value[C][i] = value[A][i]*value[B][i];
	}
}

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

	s=1;
	s <<= count_bit_size;
	while( s>>=1 ) tmp |= r & s;

	return tmp;
}

void copy_with_bit_inverse( int array )
{
	unsigned int i;
	unsigned int inverse;

	for( i=0 ; i<poly_size[A] ; i++ )
	{
		inverse = inverse_bits( i );
		__real__(value[array][inverse]) = poly[array][i];
	}
}

void fft( int array )
{
	int block_size;
	complex tmp_omega,tmp1,tmp2;
	int i,j;

	copy_with_bit_inverse( array );

	block_size=1;
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
		block_size = 2*block_size;
	}
}

void calculate_omega( void )
{
	int i;

	omega[0]=1;

	for( i=1 ; i<C_SIZE ; i++ )
	{
		omega[i]=omega[i-1] * cexp(8*atan(1.0)*I/C_SIZE);
	}
}

void inverse_fft()
{
	int block_size;
	complex tmp_omega,tmp1,tmp2;
	int i,j;
	int array=2;

	for( i=0 ; i<C_SIZE ; i++ )
	{
		value[2][i] = conj(value[2][i]);
	}

	block_size=1;
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
		block_size = 2*block_size;
	}

	for( i=0 ; i<C_SIZE ; i++ )
	{
		value[2][i] = value[2][i]/C_SIZE;
	}

	unsigned int k;
	unsigned int inverse;

	for( k=0 ; k<C_SIZE ; k++ )
	{
		inverse = inverse_bits( k );
		poly[array][inverse] = __real__ value[array][k];
	}
}

#if defined SERIAL
void calculate_serial( void )
{
	int i=C_SIZE;
	while( i>>=1 ) count_bit_size++;

	alloc_arrays();

	calculate_omega();

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
