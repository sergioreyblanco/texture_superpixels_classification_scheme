
/**
			  * @file				trainpredict.h
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Training and testing phases with the SVM engine.
			  */

#ifndef SVM_TRAINPREDICT_H
#define SVM_TRAINPREDICT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <cblas.h>
#include "./svm.h"
#include "../utility/data_structures.h"
#include "../utility/preprocess.h"
#include "../utility/postprocess.h"
#include "../utility/general_utilities.h"
#include "../texture/texture_pipelines.h"


#define PREDEFINED_BLOCK 25 //size in pixels of the prediction block




/******************************************** TRAIN **************************************/

/**
				 * @brief      Reads one line of the text training file
				 *
				 * @param      input  Textfile to read from
         * @param      line  Line data
         * @param      max_line_len Maximun expected size of the line
				 *
				 * @return     String of the line read
				 */
char* readline_train(FILE *input, char *line, int max_line_len);


/**
				 * @brief      Loads the classification problem in an appropiate data structure
				 *
				 * @param      train_image  Hyperspectral image only with the labeled pixels in GT train
         * @param      gt_train_image  Labels distinct from zero
         * @param      instances Number of elements in the problem
         * @param      prob Data structure containing the classification problem
         * @param      X_matrix Data structure containing the predictor variables of the classification problem
				 *
				 * @return     -
				 */
void load_problem_txt(const char *filename, struct svm_node * X_matrix, struct svm_problem* prob);


/**
				 * @brief      Loads the pixel classification problem in an appropiate data structure
				 *
				 * @param      train_image  Hyperspectral image only with the labeled pixels in GT train
         * @param      gt_train_image  Labels distinct from zero
         * @param      instances Number of elements in the problem
         * @param      prob Data structure containing the classification problem
         * @param      X_matrix Data structure containing the predictor variables of the classification problem
				 *
				 * @return     -
				 */
short load_problem_hsi(image_struct *train_image, reference_data_struct *gt_train_image,int instances, struct svm_problem* prob, struct svm_node *X_matrix);


/**
				 * @brief      Loads the texture classification problem in an appropiate data structure
				 *
				 * @param      descriptors  Texture arrays created with a texture pipeline
         * @param      prob Data structure containing the classification problem
         * @param      X_matrix Data structure containing the predictor variables of the classification problem
				 *
				 * @return     -
				 */
short load_problem_texture(texture_struct *descriptors, struct svm_problem* prob, struct svm_node *X_matrix);


/**
				 * @brief      Does the CV for hyperparam estimation
				 *
				 * @param      nr_fold  Number of folds of the CV (typically 5)
         * @param      param  SVM paramters data structure
         * @param      prob Data structure containing the classification problem
				 *
				 * @return     -
				 */
void do_cross_validation(int nr_fold, struct svm_parameter param, struct svm_problem prob);








/******************************************** PREDICT **************************************/


/**
				 * @brief      Reads one line of the text testing file
				 *
				 * @param      input  Textfile to read from
         * @param      line  Line data
         * @param      max_line_len Maximun expected size of the line
				 *
				 * @return     String of the line read
				 */
char* readline_predict(FILE *input, char *line, int max_line_len);


/**
				 * @brief      Uses the testing textfile to make predictions
				 *
				 * @param      input  Textfile to read from
				 * @param      output  Textfile to write the content into
         * @param      svm_model  SVM created in the training phase
         * @param      param SVM parameters tuned with the input command line arguments
				 *
				 * @return     -
				 */
void predict_txt(FILE *input, FILE *output, svm_model *svm_model, struct svm_parameter param);


/**
				 * @brief      Uses the testing pixels to make predictions
				 *
				 * @param      command_arguments  Textfile to read from
         * @param      param SVM parameters tuned with the input command line arguments
         * @param      svm_model  SVM created in the training phase
         * @param      image Hyperspectral image used to make predictions with the testing pixels
         * @param      gt_test Reference data structure with the labels to test aggainst the predictions
         * @param      error Error printing
         * @param      message Info printing
				 *
				 * @return     -
				 */
void predict_hsi(command_arguments_struct* command_arguments, struct svm_parameter param, struct svm_model *svm_model, image_struct *image, reference_data_struct *gt_test, char* error, char* message);


/**
				 * @brief      Uses the testing descriptors to make predictions
				 *
				 * @param      command_arguments  Input command arguments
         * @param      descriptors  Texture descriptors obtained with the texture pipelines
         * @param      svm_model  SVM created in the training phase
         * @param      seg_image Segmentation over the Hyperspectral image
         * @param      gt_test Reference data structure with the labels to test aggainst the predictions
         * @param      error Error printing
         * @param      message Info printing
				 *
				 * @return     -
				 */
void predict_texture(command_arguments_struct* command_arguments, texture_struct* descriptors, struct svm_model *svm_model, segmentation_struct *seg_image, reference_data_struct *gt_test, char* error, char* message);




#endif
