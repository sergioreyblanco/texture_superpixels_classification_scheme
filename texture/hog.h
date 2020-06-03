
/**
			  * @file				hog.h
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      HOG algorithm for texture descriptors computation.
			  */

#ifndef HOG_H
#define HOG_H


#include <stdio.h>
#include <vl/hog.h>
#include "../utility/data_structures.h"


  /**
  				 * @brief      Computes the HOG descriptors per segment from an image
  				 *
  				 * @param      image : img multiespec
           * @param      s : segmentation performed
  				 *
  				 * @return     Descriptors data structure
  				 */
  descriptor_model_t hog_features ( image_struct * image, unsigned int * s, int* parameters );

#endif
