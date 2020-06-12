

#ifndef COVDET_H
#define COVDET_H


#include <stdio.h>
#include <vl/covdet.h>
#include "../utility/data_structures.h"

#define 	VL_COVDET_AA_RELATIVE_INTEGRATION_SIGMA   3
#define 	VL_COVDET_AA_PATCH_EXTENT   (3*VL_COVDET_AA_RELATIVE_INTEGRATION_SIGMA)
#define 	VL_COVDET_AA_PATCH_RESOLUTION   20


detector_model_t *covdet_keypoints ( image_struct * image, float* parameters );


#endif
