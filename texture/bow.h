
/**
			  * @file				bow.h
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Bag of Words algorithm for texture descriptors computation.
			  */

#ifndef BOW_H
#define BOW_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <lapacke.h>
#include <cblas.h>
#include <float.h>
#include <sys/time.h>
#include <exception>
#include "kmeans.h"


const int SBLOCK_BOW = 2048 ;
#define NONEMODEL 0
#define SAVEMODEL 1
#define LOADMODEL 2


/**
				 * @brief      Computes the BOW descriptors per segment using a kmeans model
				 *
				 * @param      data : img multiespec
         * @param      seg : segmentation performed
         * @param      B,H,V : number of bands, width and height
         * @param      model : kmeans model
         * @param      H1,V1 : output parameters
         * @param      K : number of kmeans centers
				 *
				 * @return     Descriptor arrays
				 */
int * bow ( unsigned int * data, unsigned int * seg, int B, int H, int V, kmeans_model_t & model, int & H1, int & V1, int & K );


#endif
