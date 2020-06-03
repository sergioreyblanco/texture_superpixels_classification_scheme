
/**
			  * @file				fishervectors.cpp
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      FV algorithm for texture descriptors computation.
			  */

#include "fishervectors.h"


// // Cuenta el numero de segmentos
// Entrada: label: las etiquetas
// Salidas: tot1:numero de segmentos
// Salidas: mi1, ma1: tamano minimo y maximo de los segmentos
template < typename S >
void numero_cuencas(S *lab, int H, int V, int *tot1, int *mi1, int *ma1)
{
    int u, *count, total=0, ma=0, mi=2147483640;

    count=(int *)calloc(H*V+1,sizeof(int));
    if(count==NULL){
      printf("No memory\n"); exit(-1);
    }

    for(u=0;u<H*V;u++)
      count[lab[u]]++;

   for(u=0;u<H*V+1;u++)
    if(count[u]>0)
      total++;

   //printf("Number of segments=%d\n",total);

   count=(int *)realloc(count,total*sizeof(int));

   for(u=0;u<total;u++){
     if(count[u]<mi && count[u]>0)
      mi=count[u];
     if(count[u]>ma)
      ma=count[u];
   }

   //printf("Segment size, min=%d, max=%d\n",mi,ma);

   *tot1=total; *ma1=ma; *mi1=mi;
}


// // Se obtienen las componentes de los segmentos
// Entrada: numSegs-numero de segmentos totales, H-ancho, V-alto, seg-segmentacion con SLIC
// Salidas: nclas-numero total de pixeles por segmento
// Salidas: clas-posiciones de cada pixel del segmento
void pixeles_por_segmento(int numSegs, int H, int V, unsigned int* seg, int **nclas_aux, int ***clas_aux)
{

  int *nclas=(int *)calloc(numSegs,sizeof(int)); //vector de tamanho igual al numero de segmentos y que contiene la cantidad de pixeles de cada uno
  int **clas=(int **)malloc(numSegs*sizeof(int*)); //vector de tam igual num de segs donde cada elemento es un nuevo vector
  if(nclas==NULL || clas==NULL) { throw std::bad_alloc() ; }

  for(int i=0;i<H*V;i++){
    nclas[seg[i]]++; //se cuenta el numero de pixeles de cada segmento (OPTIMIZABLE)
  }
  for(int i=0;i<numSegs;i++){ //en cada uno de los nseg elems, se crea un nuevo vector de tamanho el numero de pixeles que contiene esa clase
      clas[i]=(int *)malloc(nclas[i]*sizeof(int));
      if(clas[i]==NULL) { throw std::bad_alloc() ; }
  }
  memset(nclas,0,numSegs*sizeof(int)); //se vacia el vector con el num de pixeles de cada seg (POR QUE?OPTIMIZABLE?)
  for(int i=0;i<H*V;i++){
    clas[seg[i]][nclas[seg[i]]++]=i; //posicion de los pixeles de los segmentos dentro de la imagen ordenados
  }
  *nclas_aux = nclas;
  *clas_aux = clas;
}


double* fishervectors_features ( unsigned int * data, segmentation_struct * s, gmm_model_t & gmm )
{

    int min, max, numSegs;
    int *nclas, **clas;

    //numero de segmentos y tamanhos min y max (en num de pixeles)
    numero_cuencas(get_segmentation_data(s), get_segmentation_width(s), get_segmentation_height(s), &numSegs, &min, &max);

    //inicializar descriptores
    //int *out_data= new int [ numSegs*2*gmm.dimensions * gmm.centers ] ( ) ;
    double *out_data= new double [ numSegs*gmm.dimensions * gmm.centers ] ( ) ;

    //obtener vector de pixeles por segmento (sus posiciones) y de num de pixeles por segmento
    pixeles_por_segmento(numSegs, get_segmentation_width(s), get_segmentation_height(s), get_segmentation_data(s), &nclas, &clas);


    /////// Fisher enconding

    double * enc = (double*) malloc(sizeof(double) * gmm.dimensions * gmm.centers * 2);
    double *datos_aux;
    //iteracion sobre los segmentos
    for(int k=0; k<numSegs; k++){
      if(k%10000==0 || k==(numSegs-1)) printf("\tFisher segment %d/%d\n",k,numSegs); //cada 100000 segmentos se imprime un mensaje explicativo

      memset(enc, 0, sizeof(double) * gmm.dimensions * gmm.centers * 2);
      datos_aux = (double *)malloc(nclas[k] * gmm.dimensions *sizeof(double)); //vector de tamaño igual al numero de pixeles del segmento mayor y el numero de bandas de este (OPTMIZABLE)

      //para cada banda de cada pixel del segmento actual
      for(int i=0;i<nclas[k];i++)
        for(int j=0;j<gmm.dimensions;j++)
          datos_aux[i*gmm.dimensions+j] = (double) data[  clas[k][i]  *gmm.dimensions+j]; //asigna los valores espectrales del pixel actual del segmento actual a un vector auxiliar antes creado que es datos2

      /*double numTerms = */vl_fisher_encode (enc, VL_TYPE_DOUBLE,
                 gmm.means, gmm.dimensions, gmm.centers,
                 gmm.covs,
                 gmm.priors,
                 datos_aux, nclas[k],
                 VL_FISHER_FLAG_IMPROVED ) ;

      //for(int j=0; j<gmm.dimensions * gmm.centers*2; j++)
      for(int j=0; j<gmm.dimensions * gmm.centers; j++)
        //out_data[k *gmm.dimensions*gmm.centers*2 + j] = 10000*enc[ j ];
        out_data[k *gmm.dimensions*gmm.centers + j] = (double)(10000*((enc[ 2*j ] + enc[ 2*j+1 ])/2));

      free(datos_aux);
    }


    free(enc);

   return  out_data  ;
}
