
/**
			  * @file				general_utilities.h
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Very general utilities needed: time couting, sorting, etc.
			  */

#ifndef GENERAL_UTILITIES_H
#define GENERAL_UTILITIES_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>


// text printing formatting
#define BOLD "\x1B[1m" //bolded letters
#define UNDERLINED "\x1B[4m" //underlined words
#define BLINK "\x1B[5m" //blinking printing
#define HIGHLIGHTED "\x1B[7m" //highlights the printing
#define RED "\x1B[31m" // red color
#define GREEN "\x1B[32m" // green color
#define RESET "\x1B[0m" // returns to the normal printing


const char help_message[5000] =
"Usage: ./classification_scheme [hyperspectral image] [train reference data] [test reference data] [options]\n"
"\n\toptions:\n\n"
"\t-s  -->  input_seg : input segmented image in RAW format | DEFAULT = segmentation algorithm applied to hyperspectral image\n"
"\t-m  -->  output_clasfmap : output classification map | DEFAULT = output/map.ppm\n"
"\t-f  -->  output_clasftxt : output classification textfile | DEFAULT = output/prediction.txt\n"
"\t-p  -->  trainpredict_type : type of train and prediction procedure | DEFAULT = 3\n"
"\t\t1 -- training by pixel and testing by pixel\n"
"\t\t2 -- training by pixel and testing by blocks\n"
"\t\t3 -- training by segments and testing by segments (most useful and fast mode)\n"
"\t-k  -->  kernel_type : SVM kernel type | DEFAULT = 0\n"
"\t\t0 -- LINEAR kernel\n"
"\t\t1 -- POLYNOMIAL kernel\n"
"\t\t2 -- RBF kernel\n"
"\t\t3 -- SIGMOID kernel\n"
"\t-c  -->  C : set the parameter C of C-SVC | DEFAULT = 0.02\n"
"\t-o  -->  output_model : output SVM model generated in train phase | DEFAULT = output/output.model\n"
"\t-v  -->  verbose : set the quiet or verbose mode | DEFAULT = true\n"
"\t-t  -->  texture_pipeline : texture algorithms (pipeline to use) | DEFAULT = 0\n"
"\t\t0 -- No texture algorithms\n"
"\t\t1  -- Kmeans + VLAD\n"
"\t\t2  -- Kmeans + BoW\n"
"\t\t3  -- GMM + FisherVectors\n"
"\t\t4  -- SIFT + Kmeans + VLAD\n"
"\t\t5  -- SIFT + GMM + FisherVectors\n"
"\t\t6  -- SIFT + Kmeans + VLAD (descriptors)\n"
"\t\t7  -- SIFT + GMM + FisherVectors (descriptors)\n"
"\t\t8  -- DSIFT + Kmeans + VLAD\n"
"\t\t9  -- DSIFT + GMM + FisherVectors\n"
"\t\t10 -- DSIFT + Kmeans + VLAD (descriptors)\n"
"\t\t11 -- DSIFT + GMM + FisherVectors (descriptors)\n"
"\t\t12 -- LIOP + Kmeans + VLAD\n"
"\t\t13 -- LIOP + GMM + FisherVectors\n"
"\t\t14 -- LIOP + Kmeans + VLAD (descriptors)\n"
"\t\t15 -- LIOP + GMM + FisherVectors (descriptors)\n"
"\t\t16 -- HOG + Kmeans + VLAD\n"
"\t\t17 -- HOG + GMM + FisherVectors\n"
"\t\t18 -- HOG + Kmeans + VLAD (descriptors)\n"
"\t\t19 -- HOG + GMM + FisherVectors (descriptors)\n"
"\t\t20 -- LBP + Kmeans + VLAD\n"
"\t\t21 -- LBP + GMM + FisherVectors\n"
"\t\t22 -- LBP + Kmeans + VLAD (descriptors)\n"
"\t\t23 -- LBP + GMM + FisherVectors (descriptors)\n"
"\t\t24 -- MSER&SIFT + Kmeans + VLAD\n"
"\t\t25 -- MSER&SIFT + GMM + FisherVectors\n"
"\t\t26 -- MSER&SIFT + Kmeans + VLAD (descriptors)\n"
"\t\t27 -- MSER&SIFT + GMM + FisherVectors (descriptors)\n"
"\t-4  -->  sift_thresholds : 2 thresholds for the SIFT algorithm | DEFAULT = 0.1 2.5\n"
"\t\t* peak_threshold : filters peaks of the DoG scale space that are too small (in absolute value).\n"
"\t\t* edge_threshold : eliminates peaks of the DoG scale space whose curvature is too small.\n"
"\t-7  -->  dsift_parameters : 4 parameters for the DSIFT algorithm | DEFAULT = 2 4 4 8\n"
"\t\t* bin_size : size of the spatial bins in the X and Y planes.\n"
"\t\t* num_bin_X : number of spatial bins in the X plane.\n"
"\t\t* num_bin_Y : number of spatial bins in the Y plane.\n"
"\t\t* num_bin_T : number of orientation bins.\n"
"\t-12 -->  LIOP_parameters : 5 parameters for the LIOP algorithm | DEFAULT = 11 2 5 2 0.1\n"
"\t\t* side_length : width of the input image patch (the patch is square).\n"
"\t\t* num_neighbours	: number of neighbours.\n"
"\t\t* num_spatial_bins : number of spatial bins.\n"
"\t\t* radius : radius of the circular sample neighbourhoods.\n"
"\t\t* intensity_threshold : non-negative, the threshold as is is used when comparing intensities; negative, the absolute value of the specified number is multipled by the maximum intensity difference inside a patch to obtain the threshold.\n"
"\t-5  -->  HOG_parameters : 3 parameters for the HOG algorithm | DEFAULT = 32 8 0\n"
"\t\t* num_orientations	: number of distinguished orientations.\n"
"\t\t* cell_size : size of a HOG cell.\n"
"\t\t* bilinear_orientation_assignments : 1 if orientations should be assigned with bilinear interpolation.\n"
"\t-6  -->  LBP_parameters : 1 parameter for the LBP algorithm | DEFAULT = 100\n"
"\t\t* cellSize : size of the LBP cells.\n"
"\t-r  -->  reduction_method : mean or PCA reduction methods | DEFAULT = 1\n"
"\t\t1 -- mean\n"
"\t\t3 -- PCA\n"
"\n\n\t* Parameter values -p (1 or 2) and -t (any) are mutually exclusive. So the situations \"-p 1 -t (any)\" and \"-p 2 -t (any)\" are forbidden."
"\n\t* The \"(descriptors)\" annotation in some texture_pipelines significantly differ from the ones without this annotation";


void find_maxmin(unsigned int *data, int numData, long long unsigned int* min_value, long long unsigned int* max_value);


int factorial(int n);

/**
				 * @brief      Gets the index of an element in an array
				 *
				 * @param      array  Array to search in
				 * @param      length  Size of the array
				 * @param      element  Element to get the index in the array
				 *
				 * @return     -
				 */
int index_element(unsigned int* array, int length, unsigned int element);


/**
				 * @brief      Starts the time couting for a certain function
				 *
				 * @param      function_name  Name to print
				 *
				 * @return     -
				 */
void start_crono(const char* function_name);


/**
				 * @brief      Stops the time couting for a certain function
				 *
				 * @param      -
				 *
				 * @return     -
				 */
void stop_crono();


/**
				 * @brief      Prints information messages formatted
				 *
				 * @param      message  String to format and print
				 *
				 * @return     -
				 */
void print_info(char* message);


/**
				 * @brief      Prints error messages formatted
				 *
				 * @param      error  String to format and print
				 *
				 * @return     -
				 */
void print_error(char* message);


/**
				 * @brief      Checks if an element is inside an array
				 *
				 * @param      element  Element to check
         * @param      array  Array to search in
         * @param      length_array  Size of the previous array
				 *
				 * @return     bool depending on the presence of the element
				 */
bool not_in(int element, unsigned int* array, int length_array);


/**
				 * @brief      Sorts the elements of a numeric array
				 *
				 * @param      A  Element to check
         * @param      size  Array to search in
				 *
				 * @return     -
				 */
void sort_array(unsigned int *A, int size);


/**
				 * @brief      Gets the most frequent element in a numeric array
				 *
				 * @param      arr  Numeric array to search in
         * @param      n  Size of the numeric array
				 *
				 * @return     most frequent element
				 */
int most_frequent_element(unsigned int *arr, int n);


/**
				 * @brief      Shows the correct usage of the software
				 *
				 * @param      -
				 *
				 * @return     -
				 */
void exit_with_help();


/**
				 * @brief      Divides a value not divisible by another
				 *
				 * @param      n  number of parts to divide
         * @param      x  element to divide in n parts
				 *
				 * @return     array with the division needed
				 */
int* force_integer_splits(int n, int x);


/**
				 * @brief      Shows an error when reading a textfile (containing the problem)
				 *
				 * @param      line_num  Number of the line with the problem detected
				 *
				 * @return     -
				 */
void exit_input_error(int line_num);


#endif
