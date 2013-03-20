/*
 * =====================================================================================
 *
 *       Filename:  calculate.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  13.2.2013 23:14:57
 *       Compiler:  gcc
 *
 *         Author:  Robert David
 *
 * =====================================================================================
 */

#ifndef __calculate_h__
#define __calculate_h__

#if defined OPENMP
void calculate_openmp( void );
#endif
#if defined CUDA
void calculate_cuda( void );
#endif
#if defined SERIAL
void calculate_serial( void );
#endif

#endif
