/*
 * =====================================================================================
 *
 *       Filename:  naive.cu
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

#if defined __CUDACC__
void calculate_cuda( void )
{
	int i,j;

	for( i=0; i<poly_size[0]; i++ )
	{
		for( j=0; j<poly_size[1]; j++ )
		{
			poly[2][i+j] += poly[0][i]*poly[1][j];
		}
	}
}
#endif
