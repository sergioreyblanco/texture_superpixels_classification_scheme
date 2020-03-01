
/**
			  * @file				fishervectors.h
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      FV algorithm for texture descriptors computation.
			  */

#ifndef FISHERVECTORS_H
#define FISHERVECTORS_H


#include <stdlib.h>
#include <stdio.h>
#include <exception>
#include <vl/fisher.h>
#include "gmm.h"
#include "../utility/data_structures.h"

/**
         * @brief      Computes the FV descriptors per segment from an image
         *
         * @param      image : img multiespec
         * @param      s : segmentation performed
         * @param      gmm : GMM model
         *
         * @return     Descriptors data structure
         */
  int* fishervectors_features ( image_struct * image, unsigned int * s, gmm_model_t & gmm );

#endif
