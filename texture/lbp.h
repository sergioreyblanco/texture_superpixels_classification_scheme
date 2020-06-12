
/**
			  * @file				lbp.h
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      LBP algorithm for texture descriptors computation.
			  */

#ifndef LBP_H
#define LBP_H

#include <stdio.h>
#include <vl/lbp.h>
#include <math.h>
#include "../utility/data_structures.h"

  /**
  				 * @brief      Computes the LBP descriptors per segment from an image
  				 *
  				 * @param      image : img multiespec
           * @param      s : segmentation performed
  				 *
  				 * @return     Descriptors data structure

  				 */
           descriptor_model_t lbp_features ( image_struct * image, unsigned int * s, int parameter );
#endif
