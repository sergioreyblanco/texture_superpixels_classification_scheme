
/**
			  * @file				gmm.cpp
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      GMM algorithm for soft centroids computation.
			  */

#include "gmm.h"


gmm_model_t gmm ( unsigned int *data, gmm_parameter_t & params )
{
  gmm_model_t model ;

  printf("\tCenters: %d\n", params.centers);
  printf("\tDimensions: %d\n", params.dimensions);
  printf("\tMaxIterations: %d\n", params.maxiters);


  VlGMM * gmm = vl_gmm_new (params.dataType, params.dimensions, params.centers) ;

  vl_gmm_set_max_num_iterations (gmm, params.maxiters) ;
  vl_gmm_set_initialization (gmm, params.initializationMethod) ;

  double * dataDouble = (double*) malloc(sizeof(double) * params.dimensions * params.numPixels);
  for(int i=0; i<params.dimensions * params.numPixels; i++){
    dataDouble[i] = (double) data[i];
  }

  /*double LL = */vl_gmm_cluster(gmm, dataDouble, params.numPixels) ;

  free(dataDouble);

  double * means = (double*) malloc(sizeof(double) * params.dimensions*params.centers);
  memcpy(means, vl_gmm_get_means(gmm), sizeof(double) * params.dimensions*params.centers);
  double * covs = (double*) malloc(sizeof(double) * params.dimensions*params.centers);
  memcpy(covs, vl_gmm_get_covariances(gmm), sizeof(double) * params.dimensions*params.centers);
  double * priors = (double*) malloc(sizeof(double) * params.centers);
  memcpy(priors, vl_gmm_get_priors(gmm), sizeof(double) * params.centers);

  model.means = means;
  model.covs = covs;
  model.priors = priors;
  model.centers = params.centers;
  model.dimensions = params.dimensions;
  model.dataType = params.dataType;
  model.numPixels = params.numPixels;


  return model ;
}
