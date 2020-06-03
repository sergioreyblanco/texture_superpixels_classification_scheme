
/**
			  * @file				gmm.h
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      GMM algorithm for soft centroids computation.
			  */

#ifndef GMM_H
#define GMM_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vl/gmm.h>
#include "pca.h"

  // data structure with the parameters needed for the GMM computation
  struct gmm_parameter_t {
      int dataType = VL_TYPE_DOUBLE;
      int centers = 32;
      int dimensions = 5;
      int maxiters = 200;
      VlGMMInitialization initializationMethod = VlGMMKMeans; //VlGMMKMeans //VlGMMRand
      int numPixels = 0;
  } ;

  // GMM model obtained after computation
  struct gmm_model_t {
      double * means ;
      double * covs ;
      double * priors ;
      int dataType ;
      int centers ;
      int dimensions ;
      int numPixels ;
  } ;


  /**
           * @brief      Computes soft centroids per image
           *
           * @param      data : img multiespec
           * @param      params : data structure with the parameters needed for the GMM computation
           *
           * @return     GMM model
           */
  gmm_model_t gmm ( unsigned int *data, gmm_parameter_t & params );


#endif
