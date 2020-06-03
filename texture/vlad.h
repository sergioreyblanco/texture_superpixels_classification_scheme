
/**
			  * @file				vlad.h
			  *
				* @author			(C) 2014 Francisco Arguello, Dora B. Heras; Universidade de Santiago de Compostela
				*
			  * @brief      Algoritmo VLAD para imagenes de sensado remoto.
			  */

#ifndef VLAD_H
#define VLAD_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <lapacke.h>
#include <cblas.h>
#include <float.h>
#include <sys/time.h>
#include <exception>
#include "kmeans.h"
#include "sift.h"
#include "liop.h"

/**
         * @brief      Computes hard centroids per image
         *
         * @param      data : img multiespec
         * @param      seg : segmentacion realizada con slic
         * @param      B,H,V : numero de bandas, ancho y alto
         * @param      model : modelo kmeans
         * @param      model : kmeans output model
         * @param      H1,V1 : parametros de salida (H1 contendra el numero de semgntos)
         * @param      K : numero de centroides
         *
         * @return     -
         */
double * vlad ( unsigned int * data, unsigned int * seg, int B, int H, int V, kmeans_model_t & model, int & H1, int & V1, int & K) ;

double * vlad_sift ( descriptor_model_t desc, int B, int H, int V, kmeans_model_t & model, int & H1, int & K );

double * vlad_liop ( descriptor_model_t desc, int B, int H, int V, kmeans_model_t & model, int & H1, int & K );

#endif
