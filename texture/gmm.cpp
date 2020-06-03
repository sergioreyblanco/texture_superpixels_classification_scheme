
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

  if(params.numPixels <= 160000000){
    double * dataDouble = (double*) malloc(sizeof(double) * params.dimensions * params.numPixels);
    for(int i=0; i<params.dimensions * params.numPixels; i++){
      dataDouble[i] = (double) data[i];
    }

    /*double LL = */vl_gmm_cluster(gmm, dataDouble, params.numPixels) ;

    //vl_gmm_delete (gmm);

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
  } else{
      vl_size size=5000;
      int num_models=(params.numPixels*params.dimensions)/(size*size*params.dimensions);
      printf("\t%d\n", num_models);
      double * patch=(double*)malloc(sizeof(double)*size*size*params.dimensions);
      double * all_means = (double*) malloc(sizeof(double) * params.dimensions*params.centers*num_models);
      double * all_covs = (double*) malloc(sizeof(double) * params.dimensions*params.centers*num_models);
      double * all_priors = (double*) malloc(sizeof(double) * params.centers*num_models);
      for(int i=0; i<num_models; i++){
        for(int p1 = 0; p1 < (signed)(size*size*params.dimensions); p1++){
            patch[ p1 ] = (double) data[ i*(size*size*params.dimensions) + p1 ];
        }
        vl_gmm_cluster(gmm, patch, size*size*params.dimensions) ;
        double * means = (double*) malloc(sizeof(double) * params.dimensions*params.centers);
        memcpy(means, vl_gmm_get_means(gmm), sizeof(double) * params.dimensions*params.centers);
        double * covs = (double*) malloc(sizeof(double) * params.dimensions*params.centers);
        memcpy(covs, vl_gmm_get_covariances(gmm), sizeof(double) * params.dimensions*params.centers);
        double * priors = (double*) malloc(sizeof(double) * params.centers);
        memcpy(priors, vl_gmm_get_priors(gmm), sizeof(double) * params.centers);
        for(int j = 0; j < (signed)params.dimensions*params.centers; j++){
            all_means[ i*(params.dimensions*params.centers)+j ] = means[j];
            all_covs[ i*(params.dimensions*params.centers)+j ] = covs[j];
        }
        for(int j = 0; j < (signed)params.centers; j++){
            all_priors[ i*(params.centers)+j ] = priors[j];
        }
        free(means); free(covs); free(priors);
      }

      model.centers = params.centers;
      model.dimensions = params.dimensions;
      model.dataType = params.dataType;
      model.numPixels = params.numPixels;

      //printf("*************5\n");
      Eigen::MatrixXf pca_data_matrix_means(num_models*params.centers, params.dimensions );
      Eigen::MatrixXf pca_data_matrix_covs(num_models*params.centers, params.dimensions );
      for(int i=0; i<num_models*params.centers; i++){
        for(int j=0; j<params.dimensions; j++){
          pca_data_matrix_means(i,j) = all_means[i*params.dimensions+j];
          pca_data_matrix_covs(i,j) = all_covs[i*params.dimensions+j];
          //printf("%f  ", all_means[i*params.dimensions+j]);
        }//printf("\n\n");
      }

      Eigen::MatrixXf pca_data_matrix_priors(params.centers, num_models );
      for(int i=0; i<params.centers; i++){
        for(int j=0; j<num_models; j++){
          pca_data_matrix_priors(i,j) = all_priors[j*params.centers+i];
          //printf("%f  ", pca_data_matrix_priors(i,j));
        }//printf("\n\n");
      }

      //printf("%d %d\n", pca_data_matrix_means.rows(), pca_data_matrix_means.cols());
      //printf("%d %d\n", pca_data_matrix_priors.rows(), pca_data_matrix_priors.cols());
      /*for(int i=0;i<model.centers*num_models;i++){
        for(int j=0;j<model.dimensions;j++){
          printf("%f  ", pca_data_matrix_means(i,j));
        }printf("\n\n");
      }*/
      //printf("*************6\n");
      pca_data_matrix_means.transposeInPlace();
      pca_data_matrix_covs.transposeInPlace();
      /*printf("%d %d\n", pca_data_matrix_means.rows(), pca_data_matrix_means.cols());
      for(int i=0;i<model.dimensions;i++){
        for(int j=0;j<model.centers*num_models;j++){
          printf("%f  ", pca_data_matrix_means(i,j));
        }printf("\n\n");
      }*/
      //printf("***************7\n");
      pca_t<float> pca_means;
      pca_t<float> pca_covs;
      pca_means.set_input(pca_data_matrix_means);
      pca_means.compute();
      pca_covs.set_input(pca_data_matrix_covs);
      pca_covs.compute();
      pca_t<float> pca_priors;
      pca_priors.set_input(pca_data_matrix_priors);
      pca_priors.compute();
      //printf("8\n");
      double *model_aux_means=(double *)calloc(model.centers*model.dimensions,sizeof(double));
      double *model_aux_covs=(double *)calloc(model.centers*model.dimensions,sizeof(double));
      for(int k=0;k<model.dimensions;k++){
      	for(int i=0;i<model.centers;i++){
          model_aux_means[k*model.centers+i]=0;
          model_aux_covs[k*model.centers+i]=0;
          for(int j=0;j<model.centers*num_models;j++){
      		    model_aux_means[k*model.centers+i] = model_aux_means[k*model.centers+i] + pca_data_matrix_means(k,j)*pca_means.get_eigen_vectors()(j,i);
              model_aux_covs[k*model.centers+i] = model_aux_covs[k*model.centers+i] + pca_data_matrix_covs(k,j)*pca_covs.get_eigen_vectors()(j,i);
          }
      	}
      }
      //printf("8b\n");
      //printf("%d %d\n", pca_data_matrix_priors.rows(), pca_data_matrix_priors.cols());
      //printf("%d %d\n", pca_priors.get_eigen_vectors().rows(), pca_priors.get_eigen_vectors().cols());
      double *model_aux_priors=(double *)calloc(model.centers,sizeof(double));
      for(int k=0;k<1;k++){
      	for(int i=0;i<model.centers;i++){
          model_aux_priors[k*model.centers+i]=0;
          for(int j=0;j<model.centers;j++){
      		    model_aux_priors[k*model.centers+i] = model_aux_priors[k*model.centers+i] + pca_data_matrix_priors(k,j)*pca_priors.get_eigen_vectors()(j,i);
          }
      	}
      }
      /*for(int i=0;i<model.dimensions;i++){
        for(int j=0;j<model.centers;j++){
          printf("%f  ", model_aux[i*model.centers+j]);
        }printf("\n\n");
      }*/
      //printf("**************9\n");
      model.means=(double *)calloc(model.centers*model.dimensions,sizeof(double));
      model.covs=(double *)calloc(model.centers*model.dimensions,sizeof(double));
      for(int i=0; i<model.centers; i++){
        for(int j=0; j<model.dimensions; j++){
          model.means[i*model.dimensions+j] = model_aux_means[j*model.centers+i];
          model.covs[i*model.dimensions+j] = model_aux_covs[j*model.centers+i];
          //printf("%f  ", model.means[i*model.dimensions+j]);
        }
        //printf("\n");
      }
      free(model_aux_means);
      free(model_aux_covs);
      free(model_aux_priors);

      model.priors=(double *)calloc(model.centers,sizeof(double));
      for(int j=0; j<model.centers; j++){
        model.priors[j] = model_aux_priors[j];
      }

      for(int i=0; i<model.centers; i++){
        for(int j=0; j<model.dimensions; j++){
          //printf("%f  ", model.means[i*model.dimensions+j]);
        }
        //printf("\n");
      }
      for(int i=0; i<model.centers; i++){
        for(int j=0; j<model.dimensions; j++){
          //printf("%f  ", model.covs[i*model.dimensions+j]);
        }
        //printf("\n");
      }
      for(int i=0; i<model.centers; i++){
        //printf("%f  ", model.priors[i]);
      }
      //printf("10\n");

  }



  return model ;
}
