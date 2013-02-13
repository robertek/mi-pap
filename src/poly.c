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

#define DEBUG( string, ... ) printf( string, __VA_ARGS__ )
//#define DEBUG( ... ) 
#define ERR( string ) { fprintf( stderr, string ); safe_exit( 1 ); }

#define A 0
#define B 1
#define C 2

int * poly_size;
int ** poly;


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

int load_file( char * file )
{
	FILE * fd;
	int i,num;

	fd = fopen( file, "r" );
	if( ! fd ) return 1;

	poly_size = (int*)malloc( 3*sizeof(int) );
	poly = (int**)malloc( 3*sizeof(int*) );

	for( num=0; num<2; num++ )
	{
		fscanf( fd, "%d", &poly_size[num] );
		if( ! poly_size[num] ) return 1;
		poly_size[2] += poly_size[num];
		poly[num] = (int*)malloc( poly_size[num]*sizeof(int) );
		for( i=0; i<poly_size[num]; i++ )
		{
			if ( ! fscanf( fd, "%d", &poly[num][i] ) ) return 1;
		}
	}

	poly[2] = (int*)malloc( poly_size[2]*sizeof(int) );

	fclose( fd );
	return 0;
}

void print_output( void )
{
	int i,num;

	for( num=0; num<3; num++ )
	{
		printf( "Poly %d: ", num );
		for( i=0; i<poly_size[num]; i++ ) 
		{
			if( i != 0 ) printf( " + " );
			printf( "%dx^%d", poly[num][i], i );
		}
		printf( "\n");
	}
}

int main( int argc, char ** argv )
{
	if( argc != 2 ) ERR("No input file.\n");
	DEBUG( "Input file: %s\n", argv[1] );

	if( load_file( argv[1] ) ) ERR( "Invalid file format.\n" );

	print_output();

	safe_exit( 0 ); 
	return 0;
}
