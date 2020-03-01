
/**
			  * @file				preprocess.c
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Doing all the tasks needed before the training and predicting phases.
			  */

#ifndef PREPROCESS_H
#define PREPROCESS_H

#include "../texture/texture_pipelines.h"
#include "../texture/slic.h"
#include "./data_structures.h"
#include "./load_data.h"
#include "../svm/svm.h"


/**
				 * @brief      Null printing
				 *
				 * @param      s  String to print
				 *
				 * @return     -
				 */
void print_null(const char *s);


/**
				 * @brief      Computes the mean values of the pixels of a segment
				 *
				 * @param      seg  Segmented image data structure
         * @param      image  Hyperspectral image data structure
         * @param      number_segments Number of segments in the segmented image
				 *
				 * @return     Array with the mean values (for each band) per segment
				 */
int* get_means_per_segment(segmentation_struct* seg, image_struct* image, int number_segments);


/**
				 * @brief      Gets the reference data label that is majority in each segment
				 *
				 * @param      seg  Segmented image data structure
         * @param      gt_train  reference data structure only with the training pixels labeled
         * @param      number_segments Number of segments in the segmented image
				 *
				 * @return     -
				 */
int* get_labels_per_segment(segmentation_struct* seg, reference_data_struct* gt_train, int number_segments);


/**
				 * @brief      Removes the pixels labeled with zero in the reference data from the Hyperspectral image
				 *
				 * @param      image  Hyperspectral image data structure
         * @param      gt_image  reference data structure with its pixels labeled
         * @param      train_image Hyperspectral image data structure only with the non zero labeled pixels
				 *
				 * @return     -
				 */
void remove_unlabeled_hsi(image_struct *image, reference_data_struct *gt_image, image_struct *train_image);


/**
				 * @brief      Removes the pixels labeled with zero in the reference data from the train reference data
				 *
         * @param      gt_image  reference data structure with its pixels labeled
         * @param      gt_train_image  reference training data structure with its pixels labeled
				 *
				 * @return     -
				 */
void remove_unlabeled_gt(reference_data_struct *gt_image, reference_data_struct *gt_train_image);


/**
				 * @brief      Removes the descriptors labeled with zero in the reference data from the texture descriptors structure
				 *
         * @param      descriptors  descriptors obtained by the texture pipeline
         * @param      descriptors_train  descriptors training data structure
				 *
				 * @return     -
				 */
void remove_unlabeled_descriptors(texture_struct* descriptors, texture_struct* descriptors_train);


/**
				 * @brief      Analyzes the input command arguments and extracts the mandatory and optional parameters
				 *
				 * @param      argc  Number of parameters
         * @param      argv  Array of string with unparsed parameters
         * @param      command_arguments  Data structure with the parsed parameters
         * @param      param SVM parameters structure (will be setted inside)
         * @param      error Error printing
				 *
				 * @return     -
				 */
void parse_command_line(int argc, char **argv, command_arguments_struct* command_arguments, struct svm_parameter* param, char* error);


/**
				 * @brief      Does a segmentation process using diferente algorithm
				 *
				 * @param      algorithm  Algorith to choose
         * @param      image  Hyperspectral image data structure
         * @param      seg_image Segmented image data structure
         * @param      error Error printing
				 *
				 * @return     -
				 */
void do_segmentation(int algorithm, image_struct* image, segmentation_struct* seg_image, char* error);

#endif
