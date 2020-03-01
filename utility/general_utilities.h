
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
#include <time.h>

// text printing formatting
#define BOLD "\x1B[1m" //bolded letters
#define UNDERLINED "\x1B[4m" //underlined words
#define BLINK "\x1B[5m" //blinking printing
#define HIGHLIGHTED "\x1B[7m" //highlights the printing
#define RED "\x1B[31m" // red color
#define GREEN "\x1B[32m" // green color
#define RESET "\x1B[0m" // returns to the normal printing


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
