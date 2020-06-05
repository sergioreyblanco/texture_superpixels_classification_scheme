
/**
			  * @file				texture_pipelines.h
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Different grouping of texture algorithm for texture descriptors obtaining.
			  */

#ifndef TEXTURE_PIPELINES_H
#define TEXTURE_PIPELINES_H

#include "../utility/data_structures.h"
#include "../utility/general_utilities.h"
#include "../utility/preprocess.h"
#include "kmeans.h"
#include "vlad.h"
#include "bow.h"
#include "gmm.h"
#include "fishervectors.h"
#include "sift.h"
#include "dsift.h"
#include "liop.h"
#include "hog.h"
#include "lbp.h"


/**
         * @brief      Can select different texture pipelines to compute descriptors per image
         *
         * @param      image : img multiespec
         * @param      seg : segmentation performed
         * @param      gt_train : reference data structure to train phase
         * @param      num_pixels : number of pixels to compute their descriptors
         * @param      type : pipeline to choose
         * @param      error : Error printing
         *
         * @return     Descriptors data structure
         */
texture_struct* texture_pipeline(image_struct* image, image_struct* train_image,  segmentation_struct* seg, reference_data_struct* gt_train, int num_pixels,  command_arguments_struct *command_arguments, char* error);

#endif
