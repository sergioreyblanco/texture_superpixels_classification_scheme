
/**
			  * @file				postprocess.c
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Doing all the tasks needed after the training and predicting phases.
			  */

#include "postprocess.h"

void set_labels_per_segment(segmentation_struct* seg, int* classification_map, int* predict_labels_aux, int number_segments)
{
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
    for(int i=0;i<nclas[k];i++){
        classification_map[  clas[k][i]  ] = predict_labels_aux[k];
    }
  }
}


void classification_map_ppm(char *filename, int *img, unsigned int H, unsigned int V, char* error, char* message)
{
  unsigned int i;
  unsigned char p;
  FILE *fp;
  fp = fopen(filename, "w");
  if (fp == NULL) {
      print_error((char*)"No se puede guardar el mapa de clasificación");
      return;
  }
  fprintf(fp, "P6\n%d %d\n255\n", H, V);
  for (i = 0; i < H * V; i++) // segmentation, color aleatorio
  {
      p = (unsigned char) (32 * img[i]) % 256;
      fputc(p, fp);
      p = (unsigned char) (171 * img[i]) % 256;
      fputc(p, fp);
      p = (unsigned char) (237 * img[i]) % 256;
      fputc(p, fp);
  }
  fclose(fp);

  sprintf(message, "Saved " UNDERLINED "classification map" RESET GREEN " %s", filename);
  print_info((char*)message);
}


void prediction_textfile(int* classification_map, reference_data_struct* gt_test, char* path_file, char* error)
{

  FILE* output;

  if((output = fopen(path_file,"w")) == NULL) {
    sprintf(error, "can't open output file %s", path_file);
    print_error((char*)error);
    exit(EXIT_FAILURE);
  }

  for(unsigned int i=0; i<get_reference_data_width(gt_test)*get_reference_data_height(gt_test); i++){
    fprintf(output,"%d %d\n", get_reference_data_data(gt_test)[i], classification_map[i]);
  }

}


int* get_central_coords_per_segment(segmentation_struct* seg, int number_segments)
{
  int* central_coords = (int*)calloc(number_segments, sizeof(int));
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

    central_coords[k] = row*get_segmentation_width(seg)+col;
  }

  return(central_coords);
}

void confusion_matrix( reference_data_struct *gt_test, int* classification_map, segmentation_struct* segmentation )
{
  unsigned int *maximum_different_classes = (unsigned int*)calloc(MAXIMUM_NUMBER_CLASSES,sizeof(unsigned int));
  unsigned int *different_classes;
  int number_classes = 0, hits=0, total=0;
  int *conf_mat, *sum_per_row, *sum_per_col;
  double *OA_per_class, OA, AA=0, kappa=0;
  char conf_mat_print[2000], aux[30];
  memset(conf_mat_print, '\0', sizeof(conf_mat_print));


  //get different classes
  for(unsigned int i=0; i<get_reference_data_width(gt_test)*get_reference_data_height(gt_test); i++){
      if(not_in(get_reference_data_data(gt_test)[i], maximum_different_classes, MAXIMUM_NUMBER_CLASSES)  &&  get_reference_data_data(gt_test)[i] != 0){
        for(int k=0; k<MAXIMUM_NUMBER_CLASSES; k++){
          if((int)maximum_different_classes[k] == 0){
            maximum_different_classes[k] = get_reference_data_data(gt_test)[i];
            number_classes++;
            break;
          }
        }
      }
  }

  //swap to smaller array
  different_classes = (unsigned int*)malloc(number_classes*sizeof(unsigned int));
  for(int j=0; j<number_classes; j++){
    different_classes[j] = maximum_different_classes[j];
  }
  free(maximum_different_classes);

  //sort the class array
  sort_array(different_classes, number_classes);

  int* central_coords = get_central_coords_per_segment( segmentation,  get_segmentation_number_segments(segmentation));

  //fill the confusion matrix
  int sum=0, acum=1;
  conf_mat = ( int*)calloc(number_classes*number_classes,sizeof( int));
  for(unsigned int i=0; i<get_reference_data_width(gt_test)*get_reference_data_height(gt_test); i++){
    //if(get_reference_data_data(gt_test)[i] != 0 && not_in(i, (unsigned int*)central_coords, get_segmentation_number_segments(segmentation))) {
    if(i == (unsigned int) central_coords[0]){
      central_coords[0] = central_coords[acum]; //se pone el siguiente de primero: esto se puede hacer porque estan ordenados en este array
      acum++;
      continue;
    }

    if(get_reference_data_data(gt_test)[i] != 0) {
      int x1 = index_element(different_classes, number_classes, classification_map[i]);
      int x2 = index_element(different_classes, number_classes, get_reference_data_data(gt_test)[i]);

      conf_mat[ (x1)*number_classes+(x2) ]++;
      sum++;
    }
  }

  //compute the aggregate meassures
  sum_per_row = ( int*)calloc(number_classes,sizeof( int));
  sum_per_col = ( int*)calloc(number_classes,sizeof( int));
  OA_per_class = ( double*)calloc(number_classes,sizeof( double));
  for(int i=0; i<number_classes; i++){
    for(int j=0; j<number_classes; j++){
      sum_per_row[i] += conf_mat[ i*number_classes+j ];
      sum_per_col[i] += conf_mat[ j*number_classes+i ];
    }
  }

  // compute the OA per class
  for(int i=0; i<number_classes; i++){
    if(sum_per_row[i] != 0){
      OA_per_class[i] = (double)conf_mat[i*number_classes+i] / (double)sum_per_row[i];
    }
  }

  // compute the general OA
  for(int i=0; i<number_classes; i++){
    hits += conf_mat[i*number_classes+i];
    total += sum_per_row[i];
  }
  OA = (double)hits/(double)total;

  // compute the general AA
  for(int i=0; i<number_classes; i++){
    AA += OA_per_class[i];
  }
  AA=AA/(double)number_classes;

  // compute Kappa
  double po = OA;
  double pc = 0;
  for(int i=0; i<number_classes; i++){
    double n1=0, n2=0;
    for(int j=0; j<number_classes; j++){
      n1 += ((float)conf_mat[i*number_classes+j]/total);
      n2 += ((float)conf_mat[j*number_classes+i]/total);
    }
    pc += n1*n2;
  }
  kappa = (po-pc)/(1-pc);

  ////// printing all the previous computations
  //first row: titles
  strncat(conf_mat_print, "Clasf \\ GT\t", strlen("Clasf \\ GT\t"));
  for(int i=0; i<number_classes;i++){
    sprintf(aux, "\t%d", different_classes[i]);
    strncat(conf_mat_print, aux, strlen(aux));
  }
  strncat(conf_mat_print, "\t\tSUM\n\t  ", strlen("\t\tSUM\n\t  "));

  //second row: lines
  strncat(conf_mat_print, "------------------", strlen("-----------------"));
  for(int i=0; i<number_classes+1;i++){
    strncat(conf_mat_print, "---------", strlen("--------"));
  }
  strncat(conf_mat_print, "\n", strlen("\n"));

  //matrix rows
  for(int i=0; i<number_classes;i++){
    strncat(conf_mat_print, "\t  ", strlen("\t  "));
    sprintf(aux, "%d\t\t", different_classes[i]);
    strncat(conf_mat_print, aux, strlen(aux));
    for(int j=0; j<number_classes;j++){
      sprintf(aux, "\t%d", conf_mat[i*number_classes+j]);
      strncat(conf_mat_print, aux, strlen(aux));
    }
    strncat(conf_mat_print, "\t  |\t", strlen("\t  |\t"));
    sprintf(aux, "%d", sum_per_row[i]);
    strncat(conf_mat_print, aux, strlen(aux));
    strncat(conf_mat_print, "\n", strlen("\n"));
  }

  //new row: lines
  strncat(conf_mat_print, "\t  ------------------", strlen("\t  -----------------"));
  for(int i=0; i<number_classes+1;i++){
    strncat(conf_mat_print, "---------", strlen("--------"));
  }
  strncat(conf_mat_print, "\n", strlen("\n"));

  //new row: sum per column
  strncat(conf_mat_print, "\t  SUM\t\t ", strlen("\t  SUM\t\t "));
  for(int i=0; i<number_classes;i++){
    sprintf(aux, "\t%d", sum_per_col[i]);
    strncat(conf_mat_print, aux, strlen(aux));
  }
  sprintf(aux, "\t\t%d\n", total);
  strncat(conf_mat_print, aux, strlen(aux));

  //new row: class OA
  strncat(conf_mat_print, "\t  Class_OA\t ", strlen("\t  Class_OA\t "));
  for(int i=0; i<number_classes;i++){
    if(OA_per_class[i] != 0){
      sprintf(aux, "\t%0.4f", OA_per_class[i]);
      strncat(conf_mat_print, aux, strlen(aux));
    }else{
      strncat(conf_mat_print, "\t -", strlen("\t -"));
    }
  }
  strncat(conf_mat_print, "\t\t -\n", strlen("\t\t -\n"));

  //new row: space
  strncat(conf_mat_print, "\n", strlen("\n"));

  //new row: OA
  strncat(conf_mat_print, "\t  OA:\t", strlen("\t  OA:\t"));
  sprintf(aux,BLINK "\t %0.4f %%\n" RESET GREEN, OA*100);
  strncat(conf_mat_print, aux, strlen(aux));

  //new row: AA
  strncat(conf_mat_print, "\t  AA:\t", strlen("\t  AA:\t"));
  sprintf(aux, "\t %0.4f %%\n" , AA*100);
  strncat(conf_mat_print, aux, strlen(aux));

  //new row: kappa
  strncat(conf_mat_print, "\t  Kappa:", strlen("\t  Kappa:"));
  sprintf(aux, "\t %0.4f %%" , kappa*100);
  strncat(conf_mat_print, aux, strlen(aux));

  print_info((char*)conf_mat_print);
}
