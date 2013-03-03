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

#include "poly.h"


#if defined SERIAL
void calculate_serial( void )
{
}
#endif

#if defined OPENMP
void calculate_openmp( void )
{
}
#endif