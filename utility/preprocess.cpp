
/**
			  * @file				preprocess.c
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Doing all the tasks needed before the training and predicting phases.
			  */

#include "preprocess.h"


double* standardize(double* descriptors, int dim, int num, char* error)
{

  //standardize
  double* descriptors_data = (double*)malloc(dim*num*sizeof(double));
  double* mean = (double*)malloc(dim*sizeof(double));
  double* sd = (double*)malloc(dim*sizeof(double));

  for(int i=0; i<dim; i++){
    mean[i] = 0;
    for(int j=0; j<num; j++){
      mean[i] = mean[i] + descriptors[j*dim+i];
    }
    mean[i] = mean[i] / num;

    sd[i] = 0;
    for(int j=0; j<num; j++){
      sd[i] = sd[i] + (int)pow(descriptors[j*dim+i] - mean[i], 2);
    }

    sd[i] = sqrt(sd[i] * (1 / (double)(num - 1)));
  }


  for(int i=0; i<dim; i++){
    if(sd[i] != 0){
      for(int j=0; j<num; j++){
        descriptors_data[j*dim+i] = (descriptors[j*dim+i] - mean[i]) / sd[i];
      }
    } else{
      for(int j=0; j<num; j++){
        descriptors_data[j*dim+i] = descriptors[j*dim+i];
      }
    }
  }

  free(mean);
  free(sd);
  //

  return descriptors_data;
}


int* get_means_per_segment(segmentation_struct* seg, image_struct* image, int number_segments)
{
  int* means = (int*)calloc(number_segments*get_image_bands(image), sizeof(int));;
  int *nclas=(int *)calloc(number_segments,sizeof(int)); //vector de tamanho igual al numero de segmentos y que contiene la cantidad de pixeles de cada uno
  int **clas=(int **)malloc(number_segments*sizeof(int*)); //vector de tam igual num de segs donde cada elemento es un nuevo vector

  //se cuenta el numero de pixeles de cada segmento
  for(unsigned int i=0;i<get_segmentation_width(seg)*get_segmentation_height(seg);i++){
    nclas[get_segmentation_data(seg)[i]]++;
  }

  //en cada uno de los nseg elems, se crea un nuevo vector de tamanho el numero de pixeles que contiene esa clase
  for(int i=0;i<number_segments;i++){
    clas[i]=(int *)malloc(nclas[i]*sizeof(int));
  }

  //se vacia el vector con el num de pixeles de cada seg
  memset(nclas,0,number_segments*sizeof(int));


  //posicion de los pixeles de los segmentos dentro de la imagen ordenados
  for(unsigned int i=0;i<get_segmentation_width(seg)*get_segmentation_height(seg);i++){
    clas[get_segmentation_data(seg)[i]][nclas[get_segmentation_data(seg)[i]]++]=i;
  }

  //para cada segmento
  for(int k=0;k<number_segments;k++){

    // para cada pixel del segmento actual se obtienen sus valores espectrales
    //  tambien se calcula el sumatorio
    for(int i=0;i<nclas[k];i++){
      for(unsigned int j=0;j<get_image_bands(image);j++){
        means[ k*get_image_bands(image)+j ] += get_image_data(image)[  clas[k][i]*get_image_bands(image)+j  ];
      }
    }

    // se calcula la media final
    for(unsigned int j=0;j<get_image_bands(image);j++){
      means[ k*get_image_bands(image)+j ] = means[ k*get_image_bands(image)+j ] / nclas[k];
    }

  }

  return(means);
}


double* get_central_pixel_per_segment(segmentation_struct* seg, image_struct* image, int number_segments)
{
  double* central_pixels = (double*)calloc(number_segments*get_image_bands(image), sizeof(double));
  int* x1 = (int*)malloc(number_segments*2*sizeof(int));
  int* x2 = (int*)malloc(number_segments*2*sizeof(int));
  int* x3 = (int*)malloc(number_segments*2*sizeof(int));
  int* x4 = (int*)malloc(number_segments*2*sizeof(int));


  for(unsigned int k=0; k< (unsigned int)number_segments; k++){
    x1[2*k] = -1; x2[2*k] = -1; x3[2*k] = -1; x4[2*k] = -1;
    x1[2*k+1] = -1; x3[2*k+1] = -1; x3[2*k+1] = -1; x4[2*k+1] = -1;
  }


  for(unsigned int i=0; i < get_segmentation_height(seg); i++){
    for(unsigned int j=0; j < get_segmentation_width(seg); j++){
      unsigned int k = get_segmentation_data(seg)[i*get_segmentation_width(seg)+j];

      if( x1[2*k] == -1){
        x1[2*k] = (int) (i);
        x1[2*k+1] = (int) (j);
      }

      if( k != get_segmentation_data(seg)[i*get_segmentation_width(seg)+j+1]
          && ( x2[2*k]==-1 || (x2[2*k+1] < (int)j) )){
        x2[2*k] = (int) (i);
        x2[2*k+1] = (int) (j);
      }

      if( (i>0 || j>0) && k != get_segmentation_data(seg)[i*get_segmentation_width(seg)+j-1]
          && ( x3[2*k]==-1 || (x3[2*k] < (int)i) )){
        x3[2*k] = (int) (i);
        x3[2*k+1] = (int) (j);
      }

      if( k != get_segmentation_data(seg)[i*get_segmentation_width(seg)+j+1]
          && ( x4[2*k]==-1 || (x4[2*k] < (int)i) )) {
        x4[2*k] = (int) (i);
        x4[2*k+1] = (int) (j);
      }
    }
  }


  for(int k=0; k<number_segments; k++){
    //printf("\n*%d: \n", k);
    //printf("%d,%d - %d,%d - %d,%d - %d,%d\n", x1[2*k],x1[2*k+1], x2[2*k],x2[2*k+1], x3[2*k],x3[2*k+1], x4[2*k],x4[2*k+1]);

    int row=0;
    int row1 = (x3[2*k] + x1[2*k]) / 2; int row2 = (x4[2*k] + x2[2*k]) / 2;
    if(row1 == row2){
      row = row1;
    } else{
      row = (row1 + row2)/2;
    }

    int col;
    int col1 = (x2[2*k+1] + x1[2*k+1]) / 2; int col2 = (x4[2*k+1] + x3[2*k+1]) / 2;
    if(col1 == col2){
      col = col1;
    } else{
      col = (col1 + col2)/2;
    }

    //printf("%d,%d \n", row, col);
    for(unsigned int b=0; b<get_image_bands(image); b++){
      central_pixels[ k*get_image_bands(image) + b ] = (double)get_image_data(image)[ (row * get_segmentation_width(seg) + col)*get_image_bands(image) + b ];
    }
  }

  /*for(unsigned int k=0; k< (unsigned int)number_segments; k++){
    if(k==231){printf("\n*%d: \n", k);
    printf("%d,%d - %d,%d - %d,%d - %d,%d\n\n", x1[2*k],x1[2*k+1], x2[2*k],x2[2*k+1], x3[2*k],x3[2*k+1], x4[2*k],x4[2*k+1]);}
  }*/


  return(central_pixels);
}


void remove_unlabeled_descriptors(texture_struct* descriptors, texture_struct* descriptors_train)
{

    char error[100];
    unsigned int partial_index1=0, partial_index2=0;

    set_descriptors_number_descriptors(descriptors_train, get_descriptors_instances(descriptors), error);
    set_descriptors_dim_descriptors(descriptors_train, get_descriptors_dim_descriptors(descriptors), error);


    double* datos_aux = (double*)malloc(get_descriptors_instances(descriptors)*get_descriptors_dim_descriptors(descriptors)*sizeof(double));
    int* labels_aux = (int*)malloc(get_descriptors_instances(descriptors)*sizeof(int));

    for(int i=0;i<get_descriptors_number_descriptors(descriptors);i++){
      if(get_descriptors_labels(descriptors)[i] != 0){

        labels_aux[partial_index1] = (int)get_descriptors_labels(descriptors)[i];
        partial_index1++;
        for(int j=0;j<get_descriptors_dim_descriptors(descriptors);j++){
          datos_aux[partial_index2] = (double)get_descriptors_data(descriptors)[i*get_descriptors_dim_descriptors(descriptors)+j];
          partial_index2++;
        }
      }
    }

    set_descriptors_data(descriptors_train, datos_aux, error);
    free(datos_aux);

    set_descriptors_labels(descriptors_train, labels_aux, error);
    free(labels_aux);

    /*printf("%d %d %d\n", get_descriptors_dim_descriptors(descriptors_train),get_descriptors_number_descriptors(descriptors_train),get_descriptors_instances(descriptors_train));
    for(int i=0;i<get_descriptors_number_descriptors(descriptors_train);i++){
      for(int j=0;j<get_descriptors_dim_descriptors(descriptors_train);j++){
        printf("%d  ", get_descriptors_data(descriptors_train)[i*get_descriptors_dim_descriptors(descriptors_train)+j]);
      }
      printf(" -> %d\n", get_descriptors_labels(descriptors_train)[i]);
    }*/
}


int* get_labels_per_segment_majority_voting(segmentation_struct* seg, reference_data_struct* gt_train, int number_segments)
{
  int* labels = (int*)malloc(number_segments*sizeof(int));
  int *nclas=(int *)calloc(number_segments,sizeof(int)); //vector de tamanho igual al numero de segmentos y que contiene la cantidad de pixeles de cada uno
  int **clas=(int **)malloc(number_segments*sizeof(int*)); //vector de tam igual num de segs donde cada elemento es un nuevo vector

  //se cuenta el numero de pixeles de cada segmento
  for(unsigned int i=0;i<get_segmentation_width(seg)*get_segmentation_height(seg);i++){
    nclas[get_segmentation_data(seg)[i]]++;
  }

  //en cada uno de los nseg elems, se crea un nuevo vector de tamanho el numero de pixeles que contiene esa clase
  for(int i=0;i<number_segments;i++){
    clas[i]=(int *)malloc(nclas[i]*sizeof(int));
  }

  //se vacia el vector con el num de pixeles de cada seg
  memset(nclas,0,number_segments*sizeof(int));


  //posicion de los pixeles de los segmentos dentro de la imagen ordenados
  for(unsigned int i=0;i<get_segmentation_width(seg)*get_segmentation_height(seg);i++){
    clas[get_segmentation_data(seg)[i]][nclas[get_segmentation_data(seg)[i]]++]=i;
  }

  //para cada segmento
  int sum=0;
  for(int k=0;k<number_segments;k++){
    unsigned int* aux=(unsigned int*)malloc(nclas[k]*sizeof(unsigned int));

    //para cada pixel del segmento actual se obtiene su etiqueta
    for(int i=0;i<nclas[k];i++){
        aux[i] = get_reference_data_data(gt_train)[  clas[k][i]  ];
    }

    labels[k] = most_frequent_element(aux, nclas[k]);

    if(labels[ k ] !=0){
      sum++;
      //printf("\n*%d: \n", k);
      //printf("%d\n", central_labels[ (row * get_segmentation_width(seg) + col) ]);
    }

    free(aux);
  }

  return(labels);
}


int* get_labels_per_segment_central_pixels(segmentation_struct* seg, reference_data_struct* gt_train, int number_segments)
{
  int* central_labels = (int*)malloc(number_segments*sizeof(int));
  int* x1 = (int*)malloc(number_segments*2*sizeof(int));
  int* x2 = (int*)malloc(number_segments*2*sizeof(int));
  int* x3 = (int*)malloc(number_segments*2*sizeof(int));
  int* x4 = (int*)malloc(number_segments*2*sizeof(int));


  for(unsigned int k=0; k< (unsigned int)number_segments; k++){
    x1[2*k] = -1; x2[2*k] = -1; x3[2*k] = -1; x4[2*k] = -1;
    x1[2*k+1] = -1; x3[2*k+1] = -1; x3[2*k+1] = -1; x4[2*k+1] = -1;
  }


  for(unsigned int i=0; i < get_segmentation_height(seg); i++){
    for(unsigned int j=0; j < get_segmentation_width(seg); j++){
      unsigned int k = get_segmentation_data(seg)[i*get_segmentation_width(seg)+j];

      if( x1[2*k] == -1){
        x1[2*k] = (int) (i);
        x1[2*k+1] = (int) (j);
      }

      if( k != get_segmentation_data(seg)[i*get_segmentation_width(seg)+j+1]
          && ( x2[2*k]==-1 || (x2[2*k+1] < (int)j) )){
        x2[2*k] = (int) (i);
        x2[2*k+1] = (int) (j);
      }

      if( (i>0 || j>0) && k != get_segmentation_data(seg)[i*get_segmentation_width(seg)+j-1]
          && ( x3[2*k]==-1 || (x3[2*k] < (int)i) )){
        x3[2*k] = (int) (i);
        x3[2*k+1] = (int) (j);
      }

      if( k != get_segmentation_data(seg)[i*get_segmentation_width(seg)+j+1]
          && ( x4[2*k]==-1 || (x4[2*k] < (int)i) )) {
        x4[2*k] = (int) (i);
        x4[2*k+1] = (int) (j);
      }
    }
  }

  int sum=0;
  for(int k=0; k<number_segments; k++){

    //printf("%d,%d - %d,%d - %d,%d - %d,%d\n", x1[2*k],x1[2*k+1], x2[2*k],x2[2*k+1], x3[2*k],x3[2*k+1], x4[2*k],x4[2*k+1]);

    int row=0;
    int row1 = (x3[2*k] + x1[2*k]) / 2; int row2 = (x4[2*k] + x2[2*k]) / 2;
    if(row1 == row2){
      row = row1;
    } else{
      row = (row1 + row2)/2;
    }

    int col;
    int col1 = (x2[2*k+1] + x1[2*k+1]) / 2; int col2 = (x4[2*k+1] + x3[2*k+1]) / 2;
    if(col1 == col2){
      col = col1;
    } else{
      col = (col1 + col2)/2;
    }

    central_labels[ k ] = get_reference_data_data(gt_train)[ (row * get_segmentation_width(seg) + col) ];

    if(central_labels[ k ] !=0){
      sum++;
      //printf("\n*%d: \n", k);
      //printf("%d\n", central_labels[ (row * get_segmentation_width(seg) + col) ]);
    }
  }

  return(central_labels);
}


void print_null(const char *s)
{}


void remove_unlabeled_hsi(image_struct *image, reference_data_struct *gt_image, image_struct *train_image)
{

  char error[100];
  unsigned int sum=0, partial_index=0;

  for(unsigned int i=0;i<get_reference_data_width(gt_image)*get_reference_data_height(gt_image);i++){
    if(get_reference_data_data(gt_image)[i] != 0){
      sum++;
    }
  }

  set_image_path(train_image, (const char*)get_image_path(image), error);
  set_image_width(train_image, sum, error);
  set_image_height(train_image, 1, error);
  set_image_bands(train_image, get_image_bands(image), error);


  int* datos_aux = (int*)malloc(sum*get_image_bands(image)*sizeof(int));
  for(unsigned int i=0;i<get_reference_data_width(gt_image)*get_reference_data_height(gt_image);i++){
    if(get_reference_data_data(gt_image)[i] != 0){
      for(unsigned int j=0;j<get_image_bands(image);j++){
        datos_aux[partial_index*get_image_bands(image)+j] = (int)get_image_data(image)[i*get_image_bands(image)+j];
      }
      partial_index++;
    }
  }
  set_image_size(train_image,sum*get_image_bands(train_image)*sizeof(unsigned int), error);
  set_image_data(train_image, datos_aux, error);
  free(datos_aux);

  /*for(unsigned int i=0;i<sum;i++){
    for(unsigned int j=0;j<get_image_bands(train_image);j++){
      printf("%d  ",get_image_data(train_image)[i*get_image_bands(train_image)+j]);
    }  printf("\n\n\n");
  }*/

}


void remove_unlabeled_gt(reference_data_struct *gt_image, reference_data_struct *gt_train_image)
{

  char error[100];
  unsigned int sum=0, partial_index=0;

  for(unsigned int i=0;i<get_reference_data_width(gt_image)*get_reference_data_height(gt_image);i++){
    if(get_reference_data_data(gt_image)[i] != 0){
      sum++;
    }
  }

  set_reference_data_path(gt_train_image, (const char*)get_reference_data_path(gt_image), error);
  set_reference_data_width(gt_train_image, sum, error);
  set_reference_data_height(gt_train_image, 1, error);


  int* datos_aux = (int*)malloc(sum*sizeof(int));
  for(unsigned int i=0;i<get_reference_data_width(gt_image)*get_reference_data_height(gt_image);i++){
    if(get_reference_data_data(gt_image)[i] != 0){
      datos_aux[partial_index] = (int)get_reference_data_data(gt_image)[i];
      partial_index++;
    }
  }
  set_reference_data_size(gt_train_image,sum*sizeof(unsigned int), error);
  set_reference_data_data(gt_train_image, datos_aux, error);
  free(datos_aux);

  /*for(unsigned int i=0;i<sum;i++){
    printf("%d  ",get_reference_data_data(gt_train_image)[i]);
  }*/

}


void parse_command_line(int argc, char **argv, command_arguments_struct* command_arguments, struct svm_parameter* param, char* error)
{
	int i = 1;

  //params initialization
  command_arguments->input_seg[0] = -1;
  command_arguments->output_clasfmap[0] = -1;
  command_arguments->output_clasftxt[0] = -1;
  command_arguments->trainpredict_type = -1;
  param->kernel_type = -1; command_arguments->kernel_type = -1;
  param->C = -1; command_arguments->C=-1;
  command_arguments->output_model[0] = -1;
  command_arguments->verbose = -1;
  command_arguments->texture_pipeline = -1;
  command_arguments->sift_thresholds[0] = -1; command_arguments->sift_thresholds[1] = -1;
  command_arguments->dsift_parameters[0] = -1; command_arguments->dsift_parameters[1] = -1; command_arguments->dsift_parameters[2] = -1; command_arguments->dsift_parameters[3] = -1;
  command_arguments->liop_parameters[0] = -1; command_arguments->liop_parameters[1] = -1; command_arguments->liop_parameters[2] = -1; command_arguments->liop_parameters[3] = -1; command_arguments->liop_parameters[4] = -1;
  command_arguments->hog_parameters[0] = -1; command_arguments->hog_parameters[1] = -1; command_arguments->hog_parameters[2] = -1;
  command_arguments->reduction_method = -1;


  // determine filenames
  if(argc < 4)
    exit_with_help();

  strcpy(command_arguments->input_hsi, argv[i]);
  i++;
  strcpy(command_arguments->input_gttrain,argv[i]);
  i++;
  strcpy(command_arguments->input_gttest,argv[i]);
  i++;


	// parse options
	for(i=4;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();


		switch(argv[i-1][1])
		{
      case 'h':
        print_info((char*)help_message);
        break;
      case 's':
        strcpy(command_arguments->input_seg, argv[i]);
        break;
      case 'm':
        strcpy(command_arguments->output_clasfmap, argv[i]);
        break;
      case 'f':
        strcpy(command_arguments->output_clasftxt, argv[i]);
        break;
      case 'p':
        // by pixel=1, by blocks=2, by segments=3
        command_arguments->trainpredict_type = atoi(argv[i]);
        break;
      case 'k':
        // LINEAR=0, POLY=1, RBF=2, SIGMOID=3
        param->kernel_type = atoi(argv[i]); command_arguments->kernel_type = param->kernel_type;
        break;
      case 'c':
        // good value choice = 0.02
        param->C = atof(argv[i]); command_arguments->C = param->C;
        break;
      case 'o':
        strcpy(command_arguments->output_model, argv[i]);
        break;
      case 'v':
        //1=all, 2=only texture algorithms, 3=anything
        command_arguments->verbose = atoi(argv[i]);
        if(command_arguments->verbose > 1){
          svm_set_print_string_function(&print_null);
        }
        break;
      case 't':
        //from 0 to 9 methods
        command_arguments->texture_pipeline = atoi(argv[i]);
        break;
      case '4':
        command_arguments->sift_thresholds[0] = atof(argv[i]);
        i++;
        command_arguments->sift_thresholds[1] = atof(argv[i]);

        break;
      case '7':
        command_arguments->dsift_parameters[0] = atoi(argv[i]);
        i++;
        command_arguments->dsift_parameters[1] = atoi(argv[i]);
        i++;
        command_arguments->dsift_parameters[2] = atoi(argv[i]);
        i++;
        command_arguments->dsift_parameters[3] = atoi(argv[i]);
        break;
      case '1':
        if(argv[i-1][2] != '2'){
          print_error((char*)"Unknown option");
          exit_with_help();
        }
        command_arguments->liop_parameters[0] = atoi(argv[i]);
        i++;
        command_arguments->liop_parameters[1] = atoi(argv[i]);
        i++;
        command_arguments->liop_parameters[2] = atoi(argv[i]);
        i++;
        command_arguments->liop_parameters[3] = atoi(argv[i]);
        i++;
        command_arguments->liop_parameters[4] = atof(argv[i]);
        break;
      case '5':
        command_arguments->hog_parameters[0] = atoi(argv[i]);
        i++;
        command_arguments->hog_parameters[1] = atoi(argv[i]);
        i++;
        command_arguments->hog_parameters[2] = atoi(argv[i]);
        break;

      case 'r':
        command_arguments->reduction_method = atoi(argv[i]);
        break;

			/*case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'b':
				param.probability = atoi(argv[i]);
				break;
			case 'v':
				cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
          sprintf(error, "n-fold cross validation: n must >= 2");
          print_error((char*)error);
					exit_with_help();
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;*/
			default:
        sprintf(error, "Unknown option: -%c", argv[i-1][1]);
        print_error((char*)error);
				exit_with_help();
		}
	}

  // mutually exclusive parameters
  if( (command_arguments->trainpredict_type == 1 || command_arguments->trainpredict_type == 2) && command_arguments->texture_pipeline != -1){
    print_error((char*)"Parameters *trainpredict_type* and *texture_pipeline* are mutually exclusive");
    exit_with_help();
  }

  // default values
	param->svm_type = C_SVC;
	param->degree = 3;
	param->gamma = 0;	// 1/num_features
	param->coef0 = 0;
	param->nu = 0.5;
	param->cache_size = 100;
	param->C = 1;
	param->eps = 1e-3;
	param->p = 0.1;
	param->shrinking = 1;
	param->probability = 0;
	param->nr_weight = 0;
	param->weight_label = NULL;
	param->weight = NULL;

  if(command_arguments->output_clasfmap[0] == -1){
    strcpy(command_arguments->output_clasfmap, "output/map.ppm");
  }
  if(command_arguments->output_clasftxt[0] == -1){
    strcpy(command_arguments->output_clasftxt, "output/prediction.txt");
  }
  if(command_arguments->trainpredict_type == -1){
    command_arguments->trainpredict_type = 3;
  }
  if(param->C == -1){
    param->C = 0.02; command_arguments->C = 0.02;
  }
  if(param->kernel_type == -1){
    param->kernel_type = 0; command_arguments->kernel_type = 0;
  }
  if(command_arguments->verbose == -1){
    command_arguments->verbose = 2;
    svm_set_print_string_function(&print_null);
  }
  if(command_arguments->texture_pipeline == -1){
    command_arguments->texture_pipeline = 0;
  }
  if(command_arguments->sift_thresholds[0] == -1){
    command_arguments->sift_thresholds[0] = 0.1; command_arguments->sift_thresholds[1] = 2.5;
  }
  if(command_arguments->dsift_parameters[0] == -1){
    command_arguments->dsift_parameters[0] = 2; command_arguments->dsift_parameters[1] = 4; command_arguments->dsift_parameters[2] = 4; command_arguments->dsift_parameters[3] = 8;
  }
  if(command_arguments->liop_parameters[0] == -1){
    command_arguments->liop_parameters[0] = 11; command_arguments->liop_parameters[1] = 2; command_arguments->liop_parameters[2] = 5; command_arguments->liop_parameters[3] = 2; command_arguments->liop_parameters[4] = 0.1;
  }
  if(command_arguments->hog_parameters[0] == -1){
    command_arguments->hog_parameters[0] = 32; command_arguments->hog_parameters[1] = 8; command_arguments->hog_parameters[2] = VL_FALSE;
  }
  if(command_arguments->reduction_method == -1){
    command_arguments->reduction_method = 1;
  }
}


void do_segmentation(int algorithm, image_struct* image, segmentation_struct* seg_image, char* error)
{
  int number_segments;

  switch(algorithm){
    case 1:{ // ALGORITMO 1: SLIC

      start_crono("SLIC");

      slic_parameter_t slic_params;
      slic_params.S = 10 ;   slic_params.m = 2;  slic_params.minsize = 10 ;  slic_params.CONN = 4 ;  slic_params.threshold = 0.0001 ;
      int * labels = slic (get_image_data(image) , get_image_width(image), get_image_height(image), get_image_bands(image), slic_params, &number_segments);
      load_segmentation_algorithm(seg_image, labels, get_image_width(image), get_image_height(image), error);
      set_segmentation_algorithm(seg_image, "SLIC", error);
      set_segmentation_number_segments(seg_image, number_segments, error);

      stop_crono();

      break;}

    default:{
      print_error((char*)"Segmentation algorithm not recognized");
      exit(EXIT_FAILURE);}
  }
}
