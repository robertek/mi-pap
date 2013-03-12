/*
 * =====================================================================================
 *
 *       Filename:  poly.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  13.2.2013 12:29:56
 *       Compiler:  gcc,nvcc
 *
 *         Author:  Robert David
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <stdio.h>

#include "poly.h"
#include "calculate.h"

/*
 * Exit program with all the cleaning.
 */
void safe_exit( int val )
{
	if( poly )
	{
		if( poly[A] ) free( poly[A] );
		if( poly[B] ) free( poly[B] );
		if( poly[C] ) free( poly[C] );
		free( poly );
	}

	if( poly_size ) free( poly_size );

	exit( val );
}

/*
 * Parse input file.
 */
int load_file( char * file )
{
	FILE * fd;
	int i,num;

	fd = fopen( file, "r" );
	if( ! fd ) return 1;

	poly_size = (int*)calloc( sizeof(int), 3 );
	poly = (int**)calloc( sizeof(int*), 3 );

	for( num=0; num<2; num++ )
	{
		fscanf( fd, "%d", &poly_size[num] );
		if( ! poly_size[num] ) return 1;

		poly[num] = (int*)calloc( sizeof(int), poly_size[num] );
		for( i=0; i<poly_size[num]; i++ )
		{
			if ( ! fscanf( fd, "%d", &poly[num][i] ) ) return 1;
		}
	}

	poly_size[2] = poly_size[0] + poly_size[1] - 1;
	poly[2] = (int*)calloc( sizeof(int), poly_size[2] + 1 );

	fclose( fd );
	return 0;
}

/*
 * Print results.
 */
void print_output( void )
{
	int i,num;

	for( num=0; num<3; num++ )
	{
#ifdef DEBUG
		DEBUG( "Poly %d:", num );
#endif
		for( i=0; i<poly_size[num]; i++ ) 
		{
#ifdef DEBUG
			if( i != 0 ) DEBUG( " +", 0 );
#endif
			printf( " %d", poly[num][i] );
#ifdef DEBUG
			DEBUG( "x^%d", i );
#endif
		}
		printf( "\n");
	}
}

/*
 * Main. Check and parses input, calculate and print.
 */
int main( int argc, char ** argv )
{
	if( argc != 2 ) ERR("No input file.\n");
#ifdef DEBUG
	DEBUG( "Input file: %s\n", argv[1] );
#endif

	if( load_file( argv[1] ) ) ERR( "Invalid file format.\n" );

#if defined SERIAL
	calculate_serial();
#endif
#if defined OPENMP
	calculate_openmp();
#endif

#if defined __CUDACC__
	calculate_cuda();
#endif

	print_output();

	safe_exit( 0 ); 
	return 0;
}
