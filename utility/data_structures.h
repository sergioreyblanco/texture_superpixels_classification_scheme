
/**
			  * @file				data_structures.h
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Data structures needed at execution time: hsi, segmented image, GT, etc.
			  */

#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "./general_utilities.h"


// Input command arguments
typedef struct {
	char input_hsi[1024]; //path of the hsi image
  char input_gttrain[1024]; //path of the gt train image
  char input_gttest[1024]; //path of the gt test image
  char input_seg[1024]; //path of the segmentation image
  char output_clasfmap[1024]; //path of the classification map output
  char output_clasftxt[1024]; //path of the textfile with the predictions output
  int trainpredict_type; //type of train and predictio: by pixels, by blocks or using textures
  char output_model[1024]; //path of the SVM model output
  int verbose; //choose the level of verbosity
  int kernel_type; //type of SVM kernel
  double C; //penalty parameter of SVM
	int texture_pipeline; //type of texture pipeline to use
	float sift_thresholds[2]; //thresholds for the SIFT algorithm
	int dsift_parameters[4]; //tuning parameters for DSIFT algorithm
	float liop_parameters[5]; //tuning parameters for LIOP algorithm
	int hog_parameters[3]; //tuning parameters for HOG algorithm
	float mser_parameters[7]; //tuning parameters for MSER algorithm
	int reduction_method; //dimensionality reduction method in texture pipelines
	float covdet_parameters[5];


} command_arguments_struct;


// Hyperspectral image
typedef struct {
	unsigned char *path; // image path
	size_t size;  // image size in bytes
	unsigned int *data;  // pointer to image data
	unsigned int width; // image width
	unsigned int height; // image height
	unsigned int bands; // image bands

} image_struct;


// Reference data of the current image
typedef struct {
	unsigned char *path; // image path
	size_t size;  // image size in bytes
	unsigned int *data;  // pointer to image data
	unsigned int width; // image width
	unsigned int height; // image height

} reference_data_struct;


// Image after segmentation
typedef struct {
  size_t size;  // image size in bytes
	unsigned int width; // image width
	unsigned int height; // image height
	unsigned char *path; // image path
	unsigned char *algorithm; // algorithm used for segmentation
	unsigned int number_segments; // number of segments in the segmentation image
	unsigned int *data; // pointer to image data

} segmentation_struct;


// Descriptors generated after texture pipelines
typedef struct {
	double* data; // pointer to image data
	int* labels_per_descriptors; // labels of each descriptor
  int number_descriptors; // number of descriptors
  int dim_descriptors; // dim of the descriptors
	int instances; // number of descriptors that have non zero labels

} texture_struct;


// array with for each descriptor
struct Ds { std::vector<double> desc;};

// data about the descriptors extracted from a segmented image
struct descriptor_model_t {
		std::vector<Ds>* descriptors ;
		int num_segments;
		int* descriptors_per_segment;
		int total_descriptors;
} ;

struct detector_model_t {
	  int num_patches;
		int dim_patches;
		double* patches ; //num_patches of size dim_patches
		int* coords; //x and y coords of size 2*num_patches
} ;


/************************************  Hyperspectral image  **********************************/

short set_image_path(image_struct *image, const char *path, char *error);

unsigned char* get_image_path(const image_struct *image);

short set_image_size(image_struct *image, size_t size, char *error);

size_t get_image_size(const image_struct *image);

short set_image_data(image_struct *image, int* data, char *error);

unsigned int* get_image_data(const image_struct *image);

short set_image_width(image_struct *image, unsigned int width, char *error);

unsigned int get_image_width(const image_struct *image);

short set_image_height(image_struct *image, unsigned int height, char *error);

unsigned int get_image_height(const image_struct *image);

short set_image_bands(image_struct *image, unsigned int bands, char *error);

unsigned int get_image_bands(const image_struct *image);






/************************************  Reference data  **********************************/

short set_reference_data_path(reference_data_struct *reference_data, const char *path, char *error);

unsigned char* get_reference_data_path(const reference_data_struct *reference_data);

short set_reference_data_size(reference_data_struct *reference_data, size_t size, char *error);

size_t get_reference_data_size(const reference_data_struct *reference_data);

short set_reference_data_data(reference_data_struct *reference_data, int* data, char *error);

unsigned int* get_reference_data_data(const reference_data_struct *reference_data);

short set_reference_data_width(reference_data_struct *reference_data,  unsigned int width, char *error);

unsigned int get_reference_data_width(const reference_data_struct *reference_data);

short set_reference_data_height(reference_data_struct *reference_data,  unsigned int height, char *error);

unsigned int get_reference_data_height(const reference_data_struct *reference_data);






/************************************  Segmentation map  **********************************/

short set_segmentation_path(segmentation_struct *segmentation, const char *path, char *error);

unsigned char* get_segmentation_path(const segmentation_struct *segmentation);

short set_segmentation_size(segmentation_struct *segmentation, size_t size, char *error);

size_t get_segmentation_size(const segmentation_struct *segmentation);

short set_segmentation_data(segmentation_struct *segmentation, int* data, char *error);

unsigned int* get_segmentation_data(const segmentation_struct *segmentation);

short set_segmentation_width(segmentation_struct *segmentation, unsigned int width, char *error);

unsigned int get_segmentation_width(const segmentation_struct *segmentation);

short set_segmentation_height(segmentation_struct *segmentation, unsigned int height, char *error);

unsigned int get_segmentation_height(const segmentation_struct *segmentation);

short set_segmentation_algorithm(segmentation_struct *segmentation, const char *algorithm, char *error);

unsigned char* get_segmentation_algorithm(const segmentation_struct *segmentation);

short set_segmentation_number_segments(segmentation_struct *segmentation, unsigned int number_segments, char *error);

unsigned int get_segmentation_number_segments(const segmentation_struct *segmentation);










/************************************  Texture structure  **********************************/

short set_descriptors_number_descriptors(texture_struct *descriptors, int number_descriptors, char *error);

int get_descriptors_number_descriptors(const texture_struct *descriptors);

short set_descriptors_dim_descriptors(texture_struct *descriptors, int dim_descriptors, char *error);

int get_descriptors_dim_descriptors(const texture_struct *descriptors);

short set_descriptors_data(texture_struct *descriptors, double* data, char *error);

double* get_descriptors_data(const texture_struct *descriptors);

short set_descriptors_labels(texture_struct *descriptors, int* labels, char *error);

int* get_descriptors_labels(const texture_struct *descriptors);

int get_descriptors_instances(const texture_struct *descriptors);










/************************************  Command line arguments  **********************************/

short set_command_arguments_input_hsi(command_arguments_struct *command_arguments, char* input_hsi, char *error);

const char* get_command_arguments_input_hsi(const command_arguments_struct *command_arguments);

short set_command_arguments_input_gttrain(command_arguments_struct *command_arguments, char* input_gttrain, char *error);

const char* get_command_arguments_input_gttrain(const command_arguments_struct *command_arguments);

short set_command_arguments_input_gttest(command_arguments_struct *command_arguments, char* input_gttest, char *error);

const char* get_command_arguments_input_gttest(const command_arguments_struct *command_arguments);

short set_command_arguments_input_seg(command_arguments_struct *command_arguments, char* input_seg, char *error);

const char* get_command_arguments_input_seg(const command_arguments_struct *command_arguments);

short set_command_arguments_output_clasfmap(command_arguments_struct *command_arguments, char* output_clasfmap, char *error);

const char* get_command_arguments_output_clasfmap(const command_arguments_struct *command_arguments);

short set_command_arguments_output_clasftxt(command_arguments_struct *command_arguments, char* output_clasftxt, char *error);

const char* get_command_arguments_output_clasftxt(const command_arguments_struct *command_arguments);

short set_command_arguments_trainpredict_type(command_arguments_struct *command_arguments, int trainpredict_type, char *error);

int get_command_arguments_trainpredict_type(const command_arguments_struct *command_arguments);

short set_command_arguments_output_model(command_arguments_struct *command_arguments, char* output_model, char *error);

const char* get_command_arguments_output_model(const command_arguments_struct *command_arguments);

short set_command_arguments_verbose(command_arguments_struct *command_arguments, int verbose, char *error);

int get_command_arguments_verbose(const command_arguments_struct *command_arguments);

short set_command_arguments_kernel_type(command_arguments_struct *command_arguments, int kernel_type, char *error);

int get_command_arguments_kernel_type(const command_arguments_struct *command_arguments);

short set_command_arguments_C(command_arguments_struct *command_arguments, double C, char *error);

double get_command_arguments_C(const command_arguments_struct *command_arguments);

short set_command_arguments_texture_pipeline(command_arguments_struct *command_arguments, int texture_pipeline, char *error);

int get_command_arguments_texture_pipeline(const command_arguments_struct *command_arguments);

short set_command_arguments_sift_thresholds(command_arguments_struct *command_arguments, float* sift_thresholds, char *error);

const float* get_command_arguments_sift_thresholds(const command_arguments_struct *command_arguments);

short set_command_arguments_dsift_parameters(command_arguments_struct *command_arguments, int* dsift_parameters, char *error);

const int* get_command_arguments_dsift_parameters(const command_arguments_struct *command_arguments);

short set_command_arguments_liop_parameters(command_arguments_struct *command_arguments, float* liop_parameters, char *error);

const float* get_command_arguments_liop_parameters(const command_arguments_struct *command_arguments);

short set_command_arguments_reduction_method(command_arguments_struct *command_arguments, int reduction_method, char *error);

const int get_command_arguments_reduction_method(const command_arguments_struct *command_arguments);

short set_command_arguments_hog_parameters(command_arguments_struct *command_arguments, int* hog_parameters, char *error);

const int* get_command_arguments_hog_parameters(const command_arguments_struct *command_arguments);

short set_command_arguments_mser_parameters(command_arguments_struct *command_arguments, float* mser_parameters, char *error);

const float* get_command_arguments_mser_parameters(const command_arguments_struct *command_arguments);

short set_command_arguments_covdet_parameters(command_arguments_struct *command_arguments, float* covdet_parameters, char *error);

const float* get_command_arguments_covdet_parameters(const command_arguments_struct *command_arguments);


#endif
