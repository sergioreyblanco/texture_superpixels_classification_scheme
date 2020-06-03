
/**
			  * @file				sift.h
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      SIFT algorithm for texture descriptors computation.
			  */

#ifndef SIFT_H
#define SIFT_H

#include <stdio.h>
#include <vl/sift.h>
#include <vl/covdet.h>
#include "../utility/data_structures.h"


#define VL_PI 3.141592653589793

  /**
  				 * @brief      Computes the SIFT descriptors per segment from an image
  				 *
  				 * @param      image : img multiespec
           * @param      s : segmentation performed
  				 *
  				 * @return     Descriptors data structure
  				 */
  descriptor_model_t sift_features ( image_struct * image, unsigned int * s, float* thresholds );

  descriptor_model_t raw_sift_features ( image_struct * image, unsigned int * s, detector_model_t * keypoints, float* covdet_parameters);

#endif
