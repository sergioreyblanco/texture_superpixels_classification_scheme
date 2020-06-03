
/**
			  * @file				liop.h
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      LIOP algorithm for patch texture descriptors computation.
			  */

#ifndef LIOP_H
#define LIOP_H


#include <stdio.h>
#include <vl/liop.h>
#include "../utility/data_structures.h"


  /**
  				 * @brief      Computes the LIOP descriptors per segment from an image
  				 *
  				 * @param      image : img multiespec
           * @param      s : segmentation performed
  				 *
  				 * @return     Descriptors data structure
  				 */
  descriptor_model_t liop_features ( image_struct * image, unsigned int * s, detector_model_t * keypoints, float* parameters );

#endif
