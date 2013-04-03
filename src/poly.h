/*
 * =====================================================================================
 *
 *       Filename:  poly.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  13.2.2013 22:43:30
 *       Compiler:  gcc,nvcc
 *
 *         Author:  Robert David
 *
 * =====================================================================================
 */

#ifndef __poly_h__
#define __poly_h__

/*
 * Some macro helpers.
 */
//#define DEBUG( string, ... ) printf( string, __VA_ARGS__ )
#define DEBUG( ... ) 
#define ERR( string ) { fprintf( stderr, string ); safe_exit( 1 ); }

/*
 * Define array aliases.
 */
#define A 0
#define B 1
#define C 2

/*
 * Array of poly sizes. 
 */
unsigned long int * poly_size;
/*
 * Multiple arrays of polynom coefficients.
 */
unsigned long int ** poly;

#endif
