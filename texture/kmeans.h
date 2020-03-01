
/**
			  * @file				kmeans.h
			  *
				* @author			(C) 2014 Francisco Arguello, Dora B. Heras; Universidade de Santiago de Compostela
				*
			  * @brief      Algoritmo de agrupamiento k-means para imagenes de sensado remoto.
			  */

#ifndef KMEANS_H
#define KMEANS_H


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <string>
#include <exception>


// data structure with the parameters needed for the kmeans computation
struct kmeans_parameter_t {
     int K = 32 ;
     int RINIT=0 ;
     int MAXIT=200 ;
     double EPS=0.001;
     int TIPO=0 ;
} ;


// kmeans output model after computation
struct kmeans_model_t {
    double * c ;
    double * ivar = NULL ;
    int B ;
    int K ;
    double GAMMA ;
    int mi ;
    int ma ;
    int tipo ;
} ;




/**
         * @brief      Computes hard centroids per image
         *
         * @param      datos : img multiespec
         * @param      N : number of pixels to compute the centroids from
         * @param      dim : number of bands per pixel
         * @param      params : data structure with the parameters needed for the kmeans computation
         * @param      model : kmeans output model
         *
         * @return     -
         */
void kmeans(unsigned int *datos, int N, int dim, struct kmeans_parameter_t params,  struct kmeans_model_t* model );


/**
         * @brief      Destroys the kmeans model previously created
         *
         * @param      model : kmeans model (output parameter)
         *
         * @return     -
         */
void destroy_kmeans_model ( kmeans_model_t & model );

#endif
