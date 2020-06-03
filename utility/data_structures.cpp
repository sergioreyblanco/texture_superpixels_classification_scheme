
/**
			  * @file				data_structures.c
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Data structures needed at execution time: hsi, segmented image, GT, etc.
			  */

#include "data_structures.h"


/************************************  Hyperspectral image  **********************************/

// path
short set_image_path(image_struct *image, const char *path, char *error) {
	//if (image->path != NULL) free(image->path);
	image->path = (unsigned char*) calloc(strlen(path) + 1, sizeof(unsigned char));
	if (image->path == NULL){
		print_error((char*)"Could not alloc memory");
		return EXIT_FAILURE;
	}
	strncpy((char*) image->path, path, strlen( path));

	return EXIT_SUCCESS;
}

unsigned char* get_image_path(const image_struct *image) {
	return image->path;
}



// size
short set_image_size(image_struct *image, size_t size, char *error) {
	image->size = size;

	return EXIT_SUCCESS;
}

size_t get_image_size(const image_struct *image) {
	return image->size;
}



// data
short set_image_data(image_struct *image, int* data, char *error) {
	//if (image->data != NULL) free(image->data);
	image->data = (unsigned int*) malloc(image->width*image->height*image->bands*sizeof(unsigned int));
	if (image->data == NULL){
		print_error((char*)"Could not alloc memory");
		return EXIT_FAILURE;
	}

	for(unsigned int i=0;i<image->width*image->height*image->bands;i++){
		image->data[i] = (unsigned int)data[i];
	}

	return EXIT_SUCCESS;
}

unsigned int* get_image_data(const image_struct *image) {
	return image->data;
}



// width
short set_image_width(image_struct *image, unsigned int width, char *error) {
	image->width = width;

	return EXIT_SUCCESS;
}

unsigned int get_image_width(const image_struct *image) {
	return image->width;
}



// height
short set_image_height(image_struct *image, unsigned int height, char *error) {
	image->height = height;

	return EXIT_SUCCESS;
}

unsigned int get_image_height(const image_struct *image) {
	return image->height;
}



// bands
short set_image_bands(image_struct *image, unsigned int bands, char *error) {
	image->bands = bands;

	return EXIT_SUCCESS;
}

unsigned int get_image_bands(const image_struct *image) {
	return image->bands;
}











/************************************  Reference data  **********************************/

// path
short set_reference_data_path(reference_data_struct *reference_data, const char *path, char *error) {
	//if (reference_data->path != NULL) free(reference_data->path);

	reference_data->path = (unsigned char*) calloc(strlen(path) + 1, sizeof(unsigned char));
	if (reference_data->path == NULL){
		print_error((char*)"Could not alloc memory");
		return EXIT_FAILURE;
	}
	strncpy((char*) reference_data->path, path, strlen( path));

	return EXIT_SUCCESS;
}

unsigned char* get_reference_data_path(const reference_data_struct *reference_data) {
	return reference_data->path;
}



// size
short set_reference_data_size(reference_data_struct *reference_data, size_t size, char *error) {
	reference_data->size = size;

	return EXIT_SUCCESS;
}

size_t get_reference_data_size(const reference_data_struct *reference_data) {
	return reference_data->size;
}



// data
short set_reference_data_data(reference_data_struct *reference_data, int* data, char *error) {
	//if (reference_data->data != NULL) free(reference_data->data);

	reference_data->data = (unsigned int*) malloc(reference_data->width*reference_data->height*sizeof(unsigned int));
	if (reference_data->data == NULL){
		print_error((char*)"Could not alloc memory");
		return EXIT_FAILURE;
	}
	for(unsigned int i=0;i<reference_data->width*reference_data->height;i++){
		reference_data->data[i] = (unsigned int)data[i];
	}

	return EXIT_SUCCESS;
}

unsigned int* get_reference_data_data(const reference_data_struct *reference_data) {
	return reference_data->data;
}



// width
short set_reference_data_width(reference_data_struct *reference_data, unsigned int width, char *error) {
	reference_data->width = width;

	return EXIT_SUCCESS;
}

unsigned int get_reference_data_width(const reference_data_struct *reference_data) {
	return reference_data->width;
}



// height
short set_reference_data_height(reference_data_struct *reference_data, unsigned int height, char *error) {
	reference_data->height = height;

	return EXIT_SUCCESS;
}

unsigned int get_reference_data_height(const reference_data_struct *reference_data) {
	return reference_data->height;
}








/************************************  Segmentation map  **********************************/

// size
short set_segmentation_size(segmentation_struct *segmentation, size_t size, char *error) {
	segmentation->size = size;

	return EXIT_SUCCESS;
}

size_t get_segmentation_size(const segmentation_struct *segmentation) {
	return segmentation->size;
}



// width
short set_segmentation_width(segmentation_struct *segmentation, unsigned int width, char *error) {
	segmentation->width = width;

	return EXIT_SUCCESS;
}

unsigned int get_segmentation_width(const segmentation_struct *segmentation) {
	return segmentation->width;
}



// height
short set_segmentation_height(segmentation_struct *segmentation, unsigned int height, char *error) {
	segmentation->height = height;

	return EXIT_SUCCESS;
}

unsigned int get_segmentation_height(const segmentation_struct *segmentation) {
	return segmentation->height;
}



// path
short set_segmentation_path(segmentation_struct *segmentation, const char *path, char *error) {
	//if (segmentation->path != NULL) free(segmentation->path);
	segmentation->path = (unsigned char*) calloc(strlen(path) + 1, sizeof(unsigned char));
	if (segmentation->path == NULL){
		print_error((char*)"Could not alloc memory");
		return EXIT_FAILURE;
	}
	strncpy((char*) segmentation->path, path, strlen( path));

	return EXIT_SUCCESS;
}

unsigned char* get_segmentation_path(const segmentation_struct *segmentation) {
	return segmentation->path;
}



// algorithm
short set_segmentation_algorithm(segmentation_struct *segmentation, const char *algorithm, char *error) {
	//if (segmentation->algorithm != NULL) free(segmentation->algorithm);
	segmentation->algorithm = (unsigned char*) calloc(strlen(algorithm) + 1, sizeof(unsigned char));
	if (segmentation->algorithm == NULL){
		print_error((char*)"Could not alloc memory");
		return EXIT_FAILURE;
	}
	strncpy((char*) segmentation->algorithm, algorithm, strlen( algorithm));

	return EXIT_SUCCESS;
}

unsigned char* get_segmentation_algorithm(const segmentation_struct *segmentation) {
	return segmentation->algorithm;
}



// data
short set_segmentation_data(segmentation_struct *segmentation, int* data, char *error) {
	//if (segmentation->data != NULL) free(segmentation->data);
	segmentation->data = (unsigned int*) malloc(segmentation->width*segmentation->height*sizeof(unsigned int));
	if (segmentation->data == NULL){
		print_error((char*)"Could not alloc memory");
		return EXIT_FAILURE;
	}
	for(unsigned int i=0;i<segmentation->width*segmentation->height;i++){
		segmentation->data[i] = (unsigned int)data[i];
	}

	return EXIT_SUCCESS;
}

unsigned int* get_segmentation_data(const segmentation_struct *segmentation) {
	return segmentation->data;
}



// number of segments
short set_segmentation_number_segments(segmentation_struct *segmentation, unsigned int number_segments, char *error) {
	segmentation->number_segments = number_segments;

	return EXIT_SUCCESS;
}

unsigned int get_segmentation_number_segments(const segmentation_struct *segmentation) {
	return segmentation->number_segments;
}











/************************************  Texture structure  **********************************/

// number of descriptors
short set_descriptors_number_descriptors(texture_struct *descriptors, int number_descriptors, char *error){
  descriptors->number_descriptors = number_descriptors;

  return EXIT_SUCCESS;
}

int get_descriptors_number_descriptors(const texture_struct *descriptors){
  return(descriptors->number_descriptors);
}



//dimension of the descriptors
short set_descriptors_dim_descriptors(texture_struct *descriptors, int dim_descriptors, char *error){
  descriptors->dim_descriptors = dim_descriptors;

  return EXIT_SUCCESS;
}

int get_descriptors_dim_descriptors(const texture_struct *descriptors){
  return(descriptors->dim_descriptors);
}



// data of the descriptors themselves
short set_descriptors_data(texture_struct *descriptors, double* data, char *error){
	descriptors->data = ( double*) malloc(descriptors->number_descriptors*descriptors->dim_descriptors*sizeof( double));
	if (descriptors->data == NULL){
		print_error((char*)"Could not alloc memory");
		return EXIT_FAILURE;
	}
	for( int i=0;i<descriptors->number_descriptors*descriptors->dim_descriptors;i++){
		descriptors->data[i] = ( double)data[i];
	}

  return EXIT_SUCCESS;
}

double* get_descriptors_data(const texture_struct *descriptors){
  return(descriptors->data);
}



// labels of the descriptors
short set_descriptors_labels(texture_struct *descriptors, int* labels_per_descriptors, char *error){
	descriptors->labels_per_descriptors = ( int*) malloc(descriptors->number_descriptors*sizeof( int));
	if (descriptors->labels_per_descriptors == NULL){
		print_error((char*)"Could not alloc memory");
		return EXIT_FAILURE;
	}
	for( int i=0;i<descriptors->number_descriptors;i++){
		descriptors->labels_per_descriptors[i] = ( int)labels_per_descriptors[i];
	}

  descriptors->instances = 0;
  for(int i=0;i<descriptors->number_descriptors;i++){
    if(descriptors->labels_per_descriptors[i]!=0){
      descriptors->instances++;
    }
  }

  return EXIT_SUCCESS;
}

int* get_descriptors_labels(const texture_struct *descriptors){
  return(descriptors->labels_per_descriptors);
}



// number of descriptors that have non zero labels
int get_descriptors_instances(const texture_struct *descriptors){
  return(descriptors->instances);
}







/************************************  Command line arguments  **********************************/

short set_command_arguments_input_hsi(command_arguments_struct *command_arguments, char* input_hsi, char *error){
	strncpy(command_arguments->input_hsi, input_hsi, strlen(input_hsi));

	return(EXIT_SUCCESS);
}

const char* get_command_arguments_input_hsi(const command_arguments_struct *command_arguments){
	return command_arguments->input_hsi;
}

short set_command_arguments_input_gttrain(command_arguments_struct *command_arguments, char* input_gttrain, char *error){
	strncpy(command_arguments->input_gttrain, input_gttrain, strlen(input_gttrain));

		return(EXIT_SUCCESS);
}

const char* get_command_arguments_input_gttrain(const command_arguments_struct *command_arguments){
	return command_arguments->input_gttrain;
}

short set_command_arguments_input_gttest(command_arguments_struct *command_arguments, char* input_gttest, char *error){
	strncpy(command_arguments->input_gttest, input_gttest, strlen(input_gttest));

		return(EXIT_SUCCESS);
}

const char* get_command_arguments_input_gttest(const command_arguments_struct *command_arguments){
	return command_arguments->input_gttest;
}

short set_command_arguments_input_seg(command_arguments_struct *command_arguments, char* input_seg, char *error){
	strncpy(command_arguments->input_seg, input_seg, strlen(input_seg));

		return(EXIT_SUCCESS);
}

const char* get_command_arguments_input_seg(const command_arguments_struct *command_arguments){
	return command_arguments->input_seg;
}

short set_command_arguments_output_clasfmap(command_arguments_struct *command_arguments, char* output_clasfmap, char *error){
	strncpy(command_arguments->output_clasfmap, output_clasfmap, strlen(output_clasfmap));

		return(EXIT_SUCCESS);
}

const char* get_command_arguments_output_clasfmap(const command_arguments_struct *command_arguments){
	return command_arguments->output_clasfmap;
}

short set_command_arguments_output_clasftxt(command_arguments_struct *command_arguments, char* output_clasftxt, char *error){
	strncpy(command_arguments->output_clasftxt, output_clasftxt, strlen(output_clasftxt));

		return(EXIT_SUCCESS);
}

const char* get_command_arguments_output_clasftxt(const command_arguments_struct *command_arguments){
	return command_arguments->output_clasftxt;
}

short set_command_arguments_trainpredict_type(command_arguments_struct *command_arguments, int trainpredict_type, char *error){
	command_arguments->trainpredict_type = trainpredict_type;

		return(EXIT_SUCCESS);
}

int get_command_arguments_trainpredict_type(const command_arguments_struct *command_arguments){
	return command_arguments->trainpredict_type;
}

short set_command_arguments_output_model(command_arguments_struct *command_arguments, char* output_model, char *error){
	strncpy(command_arguments->output_model, output_model, strlen(output_model));

		return(EXIT_SUCCESS);
}

const char* get_command_arguments_output_model(const command_arguments_struct *command_arguments){
	return command_arguments->output_model;
}

short set_command_arguments_verbose(command_arguments_struct *command_arguments, int verbose, char *error){
	command_arguments->verbose = verbose;

		return(EXIT_SUCCESS);
}

int get_command_arguments_verbose(const command_arguments_struct *command_arguments){
	return command_arguments->verbose;
}

short set_command_arguments_kernel_type(command_arguments_struct *command_arguments, int kernel_type, char *error){
	command_arguments->kernel_type = kernel_type;

		return(EXIT_SUCCESS);
}

int get_command_arguments_kernel_type(const command_arguments_struct *command_arguments){
	return command_arguments->kernel_type;
}

short set_command_arguments_C(command_arguments_struct *command_arguments, double C, char *error){
	command_arguments->C = C;

		return(EXIT_SUCCESS);
}

double get_command_arguments_C(const command_arguments_struct *command_arguments){
	return command_arguments->C;
}

short set_command_arguments_texture_pipeline(command_arguments_struct *command_arguments, int texture_pipeline, char *error){
	command_arguments->texture_pipeline = texture_pipeline;

		return(EXIT_SUCCESS);
}

int get_command_arguments_texture_pipeline(const command_arguments_struct *command_arguments){
	return command_arguments->texture_pipeline;
}

short set_command_arguments_sift_thresholds(command_arguments_struct *command_arguments, float* sift_thresholds, char *error){
	command_arguments->sift_thresholds[0] = sift_thresholds[0];
	command_arguments->sift_thresholds[1] = sift_thresholds[1];

		return(EXIT_SUCCESS);
}

const float* get_command_arguments_sift_thresholds(const command_arguments_struct *command_arguments){
	return command_arguments->sift_thresholds;
}

short set_command_arguments_dsift_parameters(command_arguments_struct *command_arguments, int* dsift_parameters, char *error){
	command_arguments->dsift_parameters[0] = dsift_parameters[0];
	command_arguments->dsift_parameters[1] = dsift_parameters[1];
	command_arguments->dsift_parameters[2] = dsift_parameters[2];
	command_arguments->dsift_parameters[3] = dsift_parameters[3];

		return(EXIT_SUCCESS);
}

const int* get_command_arguments_dsift_parameters(const command_arguments_struct *command_arguments){
	return command_arguments->dsift_parameters;
}

short set_command_arguments_liop_parameters(command_arguments_struct *command_arguments, float* liop_parameters, char *error){
	command_arguments->liop_parameters[0] = liop_parameters[0];
	command_arguments->liop_parameters[1] = liop_parameters[1];
	command_arguments->liop_parameters[2] = liop_parameters[2];
	command_arguments->liop_parameters[3] = liop_parameters[3];
	command_arguments->liop_parameters[4] = liop_parameters[4];

		return(EXIT_SUCCESS);
}

const float* get_command_arguments_mser_parameters(const command_arguments_struct *command_arguments){
	return command_arguments->mser_parameters;
}

short set_command_arguments_mser_parameters(command_arguments_struct *command_arguments, float* mser_parameters, char *error){
	command_arguments->mser_parameters[0] = mser_parameters[0];
	command_arguments->mser_parameters[1] = mser_parameters[1];
	command_arguments->mser_parameters[2] = mser_parameters[2];
	command_arguments->mser_parameters[3] = mser_parameters[3];
	command_arguments->mser_parameters[4] = mser_parameters[4];
	command_arguments->mser_parameters[5] = mser_parameters[5];
	command_arguments->mser_parameters[6] = mser_parameters[6];

		return(EXIT_SUCCESS);
}

const float* get_command_arguments_covdet_parameters(const command_arguments_struct *command_arguments){
	return command_arguments->covdet_parameters;
}

short set_command_arguments_covdet_parameters(command_arguments_struct *command_arguments, float* covdet_parameters, char *error){
	command_arguments->covdet_parameters[0] = covdet_parameters[0];
	command_arguments->covdet_parameters[1] = covdet_parameters[1];
	command_arguments->covdet_parameters[2] = covdet_parameters[2];
	command_arguments->covdet_parameters[3] = covdet_parameters[3];
	command_arguments->covdet_parameters[4] = covdet_parameters[4];

		return(EXIT_SUCCESS);
}

const float* get_command_arguments_liop_parameters(const command_arguments_struct *command_arguments){
	return command_arguments->liop_parameters;
}

short set_command_arguments_reduction_method(command_arguments_struct *command_arguments, int reduction_method, char *error){
	command_arguments->reduction_method = reduction_method;

		return(EXIT_SUCCESS);
}

const int get_command_arguments_reduction_method(const command_arguments_struct *command_arguments){
	return command_arguments->reduction_method;
}

short set_command_arguments_hog_parameters(command_arguments_struct *command_arguments, int* hog_parameters, char *error){
	command_arguments->hog_parameters[0] = hog_parameters[0];
	command_arguments->hog_parameters[1] = hog_parameters[1];
	command_arguments->hog_parameters[2] = hog_parameters[2];

		return(EXIT_SUCCESS);
}

const int* get_command_arguments_hog_parameters(const command_arguments_struct *command_arguments){
	return command_arguments->hog_parameters;
}
