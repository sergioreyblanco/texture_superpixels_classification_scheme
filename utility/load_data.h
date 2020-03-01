
/**
			  * @file				load_data.c
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Loading the data need in the data structures in data_structures.h.
			  */

#ifndef LOAD_DATA_H
#define LOAD_DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "./data_structures.h"
#include "./general_utilities.h"

/**
				 * @brief      Loads the hyperspectral image into memory
				 *
				 * @param      image  Data structure containing the hyperspectral image
         * @param      path  Path in disk to the hyperspectral image
         * @param      error Error printing
				 *
				 * @return     -
				 */
short load_hsi(image_struct *image, char* path, char *error);


/**
				 * @brief      Loads the reference data image into memory
				 *
				 * @param      image  Data structure containing the reference data image
         * @param      type  Type of the reference data (e.g. train, test, etc)
         * @param      path  Path in disk to the reference data image
         * @param      error Error printing
				 *
				 * @return     -
				 */
short load_gt(reference_data_struct *image, char* path, const char* type, char *error);


/**
				 * @brief      Loads the segmented image into memory
				 *
				 * @param      image  Data structure containing the segmented image
         * @param      path  Path in disk to the segmented image
         * @param      error Error printing
				 *
				 * @return     -
				 */
short load_segmentation(segmentation_struct *image, char* path, char *error);


/**
				 * @brief      Loads the segmented image into the data structure from the segmentation algorithm
				 *
				 * @param      image  Data structure that will contain the segmented image
         * @param      labels  Data of the data produced by the segmentation algorithm
         * @param      H  Width of the segmented image
         * @param      V  Height of the segmented image
         * @param      error Error printing
				 *
				 * @return     -
				 */
short load_segmentation_algorithm(segmentation_struct *image, int *labels, int H, int V, char *error);

#endif
