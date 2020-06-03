
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


const char help_message[1500] =
"Usage: ./classification_scheme [hyperspectral image] [train reference data] [test reference data] [options]\n"
"\n\toptions:\n\n"
"\t-s  -->  input_seg : input segmented image in RAW format | DEFAULT = segmentation algorithm applied to hyperspectral image\n"
"\t-m  -->  output_clasfmap : output classification map | DEFAULT = ouput/map.ppm\n"
"\t-f  -->  output_clasftxt : output classification textfile | DEFAULT = ouput/prediction.txt\n"
"\t-p  -->  trainpredict_type : type of train and prediction procedure | DEFAULT = 3\n"
"\t\t1 -- by pixel\n"
"\t\t2 -- by blocks\n"
"\t\t3 -- by segments\n"
"\t-k  -->  kernel_type : SVM kernel type | DEFAULT = 0\n"
"\t\t0 -- LINEAR kernel\n"
"\t\t1 -- POLYNOMIAL kernel\n"
"\t\t2 -- RBF kernel\n"
"\t\t3 -- SIGMOID kernel\n"
"\t-c  -->  C : set the parameter C of C-SVC | DEFAULT = 0.02\n"
"\t-o  -->  output_model : output SVM model generated in train phase | DEFAULT = output/output.model\n"
"\t-v  -->  verbose : set the quiet or verbose mode | DEFAULT = true\n"
"\t-t  -->  texture_pipeline : texture algorithms (pipeline to use) | DEFAULT = 0\n"
"\t\t0 -- no textures\n"
"\t\t1 -- kmeans + vlad\n"
"\t\t2 -- kmeans + bow\n"
"\t\t3 -- gmm + fishervectors\n"
"\t\t4 -- sift + km + vlad\n"
"\t\t5 -- sift + gmm + fishervectors\n"
"\t\t6 -- sift\n"
"\t\t7 -- dsift + km + vlad\n"
"\t\t8 -- dsift + gmm + fishervectors\n"
"\t\t9 -- dsift\n"
"\t-4  -->  sift_thresholds : 2 thresholds for the SIFT algorithm | DEFAULT = 0.1 2.5\n"
"\t-7  -->  dsift_parameters : 4 parameters for the DSIFT algorithm | DEFAULT = 2 4 4 8\n"
"\n\n\t * Parameters -t (1 or 2) and -p (any) are mutually exclusive";


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
