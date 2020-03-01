
/**
			  * @file				postprocess.c
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Doing all the tasks needed after the training and predicting phases.
			  */

#ifndef POSPROCESS_H
#define POSPROCESS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "./general_utilities.h"
#include "./data_structures.h"

#define MAXIMUM_NUMBER_CLASSES 100 //maximun number of classification categories permited by the app


/**
				 * @brief      Saves the textfile with the objective responses and the predicted ones at pixel level
				 *
         * @param      classification_map  Classification map previously computed
         * @param      gt_test Reference data data structure with the objective responses
         * @param      path_file Path of the file in disk
         * @param      error Error printing
				 *
				 * @return     -
				 */
void prediction_textfile(int* classification_map, reference_data_struct* gt_test, char* path_file, char* error);


/**
				 * @brief      Sets the predicted labels into each segment
				 *
				 * @param      seg  Segmented image data structure
         * @param      classification_map  Classification map that will be the output of the app
         * @param      predict_labels_aux Array with the predicted label for each segment
         * @param      number_segments Number of segments in the segmented image
				 *
				 * @return     -
				 */
void set_labels_per_segment(segmentation_struct* seg, int* classification_map, int* predict_labels_aux, int number_segments);


/**
				 * @brief      Saves a classification map in disk
				 *
				 * @param      filename  Path of the future image in disk
         * @param      img  Labels of each pixel of the future 2D image
         * @param      H Image width
         * @param      V Image height
         * @param      error Error printing
         * @param      message Message printing
				 *
				 * @return     -
				 */
void classification_map_ppm(char *filename, int *img, unsigned int H, unsigned int V, char* error, char* message);


/**
				 * @brief      Computes the confusion matrix and all relevant metrics
				 *
				 * @param      gt_test  Reference data image labeled only with testing pixels
         * @param      classification_map  Classification map with the output of the prediction
				 *
				 * @return     -
				 */
void confusion_matrix( reference_data_struct *gt_test, int* classification_map );

#endif
