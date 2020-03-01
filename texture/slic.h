
/**
			  * @file				slic.h
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      SLIC algorithm for segmented image computation.
			  */

#ifndef SLIC_H
#define SLIC_H



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>

  // data structure with the parameters set by the command arguments
  struct slic_parameter_t {
      int S = 50 ;
      int m = 2;
      int minsize = 90 ;
      int CONN = 4 ;
      double threshold = 0.0001 ;
  } ;

  // lab: etiqueta del segmento
  // n: numero de segmentos vecinos
  // siz: tamano de las listas vec y neig (siz>=n)
  // vec: lista de segmentos vecinos
  // weig: numero de pixeles vecinos con el segmento vecino
  // pvec: etiqueta del segmento vecino con el que vamos a fusionar
  struct obj {
    int lab;
    int siz;
    int n;
    int pvec;
    int *vec;
    int *weig;
  };

  // mezcla segmentos para conseguir un tamano minimo en base a kmeans
  // incluye ajusta_conectividad y adjust_segments
  // es el usado por defecto para todos los tipos de segmentacion
  #define OBS 20 // tamano de la lista de vecinos

  //Enough high value
  #define INF 2147483640



  /**
  				 * @brief      Computes the segmented from an hyperspectral image
  				 *
  				 * @param      img : img multiespec
           * @param      H : width of the image
           * @param      V : heiht of the image
           * @param      dim : bands of the image
           * @param      params : data structure with the parameters set by the command arguments
           * @param      number_segments : number of segments (output parameter)
  				 *
  				 * @return     Segmented image
  				 */
  int * slic (unsigned int *img, int H, int V, int dim, struct slic_parameter_t params, int* number_segments);

#endif
