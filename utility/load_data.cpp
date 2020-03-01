
/**
			  * @file				load_data.c
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Loading the data need in the data structures in data_structures.h.
			  */

#include <sys/stat.h>
#include "load_data.h"


short load_hsi(image_struct *image, char* path, char *error)
{
    FILE *fp;
    int *datos;
    int H = 0, V = 0, B = 0;
    size_t a;
    struct stat buf;
    char message[100];

    // Read file data
    set_image_path(image, path, error);
    fp = fopen((char *)get_image_path(image), "rb");
    if (fp == NULL) {
        print_error((char*)"Can not open the image");
        exit(EXIT_FAILURE);
    }

    a = fread(&B, sizeof(int), 1, fp);
    a = fread(&H, sizeof(int), 1, fp);
    a = fread(&V, sizeof(int), 1, fp);
    set_image_width(image, (unsigned int)H, error);
    set_image_height(image, (unsigned int)V, error);
    set_image_bands(image, (unsigned int)B, error);

    // Verificaciones previas
    if ( (get_image_bands(image) * get_image_width(image) * get_image_height(image)) == 0 ) {
        print_error((char*)"Incorrect image size(bands, width, height).");
        return EXIT_FAILURE;
    }

    // obtain file size
    if ( stat( (char *)get_image_path(image), &buf ) == -1 ) {
        print_error((char*)"Error getting file size.");
        return EXIT_FAILURE;
    }
    if ( ( (int) buf.st_size ) == 0 ) {
        print_error((char*)"The file is empty.");
        return EXIT_FAILURE;
    }

    // Completamos los datos de la imagen
    set_image_size(image, buf.st_size, error);
    datos = (int *) malloc(get_image_bands(image) * get_image_width(image) * get_image_height(image) * sizeof (int));
    if (datos == NULL) {
      print_error((char*)"Not enough memory\n");
      return EXIT_FAILURE;
    }
    a = fread(datos, sizeof(int), (size_t) B * H*V, fp);
    if (a != (size_t) B * H * V) {
      print_error((char*)"Read failure\n");
      return EXIT_FAILURE;
    }
    set_image_data(image, datos, error);

    free(datos);
    fclose(fp);

    sprintf(message, "Loaded " UNDERLINED "image" RESET GREEN " : B=%u, H=%u, V=%u", get_image_bands(image),get_image_height(image),get_image_width(image));
    print_info((char*)message);

    return (EXIT_SUCCESS);
}


short load_gt(reference_data_struct *gt, char* path, const char* tipo, char *error)
{
  FILE *fp;
  int *datos;
  int H = 0, V = 0;
  size_t a;
  struct stat buf;
  char message[100];

  // Read file data
  set_reference_data_path(gt, path, error);

  fp = fopen((char *)get_reference_data_path(gt), "rb");
  if (fp == NULL) {
      print_error((char*)"Can not open the ground truth.");
      exit(EXIT_FAILURE);
  }

  a = fread(&H, sizeof(int), 1, fp);
  a = fread(&H, sizeof(int), 1, fp);
  a = fread(&V, sizeof(int), 1, fp);
  set_reference_data_width(gt, (unsigned int)H, error);
  set_reference_data_height(gt, (unsigned int)V, error);

  // Verificaciones previas
  if ( (get_reference_data_width(gt) * get_reference_data_height(gt)) == 0 ) {
      print_error((char*)"Incorrect image size(width, height).");
      return EXIT_FAILURE;
  }

  // obtain file size
  if ( stat( (char *)get_reference_data_path(gt), &buf ) == -1 ) {
      print_error((char*)"Error getting file size.");
      return EXIT_FAILURE;
  }
  if ( ( (int) buf.st_size ) == 0 ) {
      print_error((char*)"The file is empty.");
      return EXIT_FAILURE;
  }

  // Completamos los datos de la imagen
  set_reference_data_size(gt, buf.st_size, error);
  datos = (int *) malloc(get_reference_data_width(gt) * get_reference_data_height(gt) * sizeof (int));
  if (datos == NULL) {
    print_error((char*)"Not enough memory\n");
    return EXIT_FAILURE;
  }
  a = fread(datos, sizeof(int), (size_t) H*V, fp);
  if (a != (size_t) H * V) {
      print_error((char*)"Read failure\n");
      return EXIT_FAILURE;
  }

  set_reference_data_data(gt, datos, error);

  free(datos);
  fclose(fp);


  sprintf(message, "Loaded " UNDERLINED "GT %s" RESET GREEN " : H=%u, V=%u", tipo, get_reference_data_height(gt),get_reference_data_width(gt));
  print_info((char*)message);

  return (EXIT_SUCCESS);
}


short load_segmentation(segmentation_struct *seg, char* path, char *error)
{
  FILE *fp;
  int *datos;
  int H = 0, V = 0;
  size_t a;
  struct stat buf;
  char message[100];

  // Read file data
  set_segmentation_path(seg, path, error);

  fp = fopen((char *)get_segmentation_path(seg), "rb");
  if (fp == NULL) {
      print_error((char*)"Can not open the segmentation.");
      exit(EXIT_FAILURE);
  }

  a = fread(&H, sizeof(int), 1, fp);
  a = fread(&V, sizeof(int), 1, fp);
  set_segmentation_width(seg, (unsigned int)H, error);
  set_segmentation_height(seg, (unsigned int)V, error);

  // Verificaciones previas
  if ( (get_segmentation_width(seg) * get_segmentation_height(seg)) == 0 ) {
      print_error((char*)"Incorrect image size(width, height).");
      return EXIT_FAILURE;
  }

  // obtain file size
  if ( stat( (char *)get_segmentation_path(seg), &buf ) == -1 ) {
      print_error((char*)"Error getting file size.");
      return EXIT_FAILURE;
  }
  if ( ( (int) buf.st_size ) == 0 ) {
      print_error((char*)"The file is empty.");
      return EXIT_FAILURE;
  }

  // Completamos los datos de la imagen
  set_segmentation_size(seg, buf.st_size, error);
  datos = (int *) malloc(get_segmentation_width(seg) * get_segmentation_height(seg) * sizeof (int));
  if (datos == NULL) {
    print_error((char*)"Not enough memory\n");
    return EXIT_FAILURE;
  }

  a = fread(datos, sizeof(int), (size_t) H*V, fp);
  if (a != (size_t) H * V) {
      print_error((char*)"Read failure\n");
      return EXIT_FAILURE;
  }
  set_segmentation_data(seg, datos, error);

  free(datos);
  fclose(fp);

  set_segmentation_algorithm(seg, "", error);
  unsigned int nsegs=0;
  for(unsigned int sg=0;sg < get_segmentation_width(seg) * get_segmentation_height(seg);sg++){
    if(get_segmentation_data(seg)[sg] > nsegs){
      nsegs = get_segmentation_data(seg)[sg];
    }
  }
  set_segmentation_number_segments(seg, nsegs+1, error);

  sprintf(message, "Loaded " UNDERLINED "segmentation" RESET GREEN " : H=%u, V=%u", get_segmentation_height(seg),get_segmentation_width(seg));
  print_info((char*)message);

  return (EXIT_SUCCESS);
}


short load_segmentation_algorithm(segmentation_struct *seg, int *labels, int H, int V, char *error)
{
  char message[100];

  set_segmentation_path(seg, "", error);

  set_segmentation_width(seg, (unsigned int)H, error);
  set_segmentation_height(seg, (unsigned int)V, error);

  // Verificaciones previas
  if ( (get_segmentation_width(seg) * get_segmentation_height(seg)) == 0 ) {
      print_error((char*)"Incorrect image size(width, height).");
      return EXIT_FAILURE;
  }

  set_segmentation_data(seg, labels, error);

  free(labels);

  sprintf(message, "Loaded " UNDERLINED "segmentation" RESET GREEN " : H=%u, V=%u", get_segmentation_height(seg),get_segmentation_width(seg));
  print_info((char*)message);

  return (EXIT_SUCCESS);
}
