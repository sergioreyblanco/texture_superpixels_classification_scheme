
/**
			  * @file				dsift.h
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      DSIFT algorithm for texture descriptors computation.
			  */

#ifndef DSIFT_H
#define DSIFT_H


#include <vector>
#include <stdio.h>
#include <vl/dsift.h>
#include "../utility/data_structures.h"
#include "sift.h"


  /**
  				 * @brief      Computes the DSIFT descriptors per segment from an image
  				 *
  				 * @param      image : img multiespec
           * @param      s : segmentation performed
  				 *
  				 * @return     Descriptors data structure
  				 */
  sift_model_t dsift_features ( image_struct * image, unsigned int * s, int* parameters );

  /**
  				 * @brief      Computes the DSIFT descriptors per segment from an image without the posibility of setting tuning parameters
  				 *
  				 * @param      image : img multiespec
           * @param      s : segmentation performed
  				 *
  				 * @return     Descriptors data structure
  				 */
  sift_model_t dsift_basic_features ( image_struct * image, unsigned int * s );

#endif
