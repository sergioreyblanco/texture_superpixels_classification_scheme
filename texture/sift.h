
/**
			  * @file				sift.h
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      SIFT algorithm for texture descriptors computation.
			  */

#ifndef SIFT_H
#define SIFT_H


#include <vector>
#include <stdio.h>
#include <vl/sift.h>
#include "../utility/data_structures.h"

  // Dimension of a SIFT descriptor
  const int dim_sift_descriptor=128;

  // array with for each descriptor
  struct Ds { vl_sift_pix desc[dim_sift_descriptor];};

  // data about the descriptors extracted from a segmented image
  struct sift_model_t {
      std::vector<Ds>* descriptors ;
      int num_segments;
      int* descriptors_per_segment;
      int total_descriptors;
  } ;


  /**
  				 * @brief      Computes the SIFT descriptors per segment from an image
  				 *
  				 * @param      image : img multiespec
           * @param      s : segmentation performed
  				 *
  				 * @return     Descriptors data structure
  				 */
  sift_model_t sift_features ( image_struct * image, unsigned int * s, float* thresholds );

#endif
