
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
short set_descriptors_data(texture_struct *descriptors, int* data, char *error){
	descriptors->data = ( int*) malloc(descriptors->number_descriptors*descriptors->dim_descriptors*sizeof( int));
	if (descriptors->data == NULL){
		print_error((char*)"Could not alloc memory");
		return EXIT_FAILURE;
	}
	for( int i=0;i<descriptors->number_descriptors*descriptors->dim_descriptors;i++){
		descriptors->data[i] = ( int)data[i];
	}

  return EXIT_SUCCESS;
}

int* get_descriptors_data(const texture_struct *descriptors){
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
