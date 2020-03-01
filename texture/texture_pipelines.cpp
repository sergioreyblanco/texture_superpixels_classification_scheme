
/**
			  * @file				texture_pipelines.cpp
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Different grouping of texture algorithm for texture descriptors obtaining.
			  */

#include "texture_pipelines.h"




texture_struct* texture_pipeline(image_struct* image,  segmentation_struct* seg, reference_data_struct* gt_train, int num_pixels, command_arguments_struct *command_arguments, char* error){
  texture_struct * descriptors = (texture_struct*)malloc(sizeof(texture_struct));
  int* data, *labels;
  int K=32;

  switch(command_arguments->texture_pipeline){
    case 0:{ // METODO 0: sin texturas (media de cada segmento)

      start_crono( "MEANS COMPUTATION" ) ;

      data = get_means_per_segment(seg, image, get_segmentation_number_segments(seg));

      stop_crono ( ) ;

      set_descriptors_dim_descriptors(descriptors, get_image_bands(image), error);
      break;}




    case 1:{ // METODO 1: Kmeans y VLAD

      kmeans_parameter_t params ;
      kmeans_model_t model ;
      int H1, V1 ;

      start_crono( "KMEANS" ) ;

      kmeans( get_image_data(image) , num_pixels, get_image_bands(image), params,  &model ) ;
      K=model.K;

      stop_crono ( ) ;



      start_crono ( "VLAD" ) ;

      data = vlad( get_image_data(image), get_segmentation_data(seg), get_image_bands(image), get_image_width(image), get_image_height(image), model, H1 , V1, K) ;

      stop_crono ( ) ;

      destroy_kmeans_model( model ) ;

      set_descriptors_dim_descriptors(descriptors, K*get_image_bands(image), error);
      break;}



    case 2:{ // METODO 2: Kmeans y BOW

      kmeans_parameter_t params ;
      kmeans_model_t model ;
      int H1, V1 ;

      start_crono( "KMEANS" ) ;

      kmeans( get_image_data(image) , num_pixels, get_image_bands(image), params,  &model ) ;

      stop_crono ( ) ;



      start_crono( "BOW" ) ;

      data = bow( get_image_data(image), get_segmentation_data(seg), get_image_bands(image), get_image_width(image), get_image_height(image), model, H1 , V1, K ) ;
      K=model.K;

      stop_crono ( ) ;

      destroy_kmeans_model( model ) ;

      set_descriptors_dim_descriptors(descriptors, K*get_image_bands(image), error);
      break;}



    case 3:{ //METODO 3: GMM y fisher vectors

      gmm_parameter_t gmm_params;

      start_crono( "GMM" ) ;
      gmm_params.dimensions = get_image_bands(image);
      gmm_params.numPixels = num_pixels;
      gmm_model_t model = gmm ( get_image_data(image), gmm_params ) ;

      stop_crono ( ) ;



      start_crono( "FisherVectors" ) ;

      data = fishervectors_features( image, get_segmentation_data(seg), model )  ;

      stop_crono ( ) ;

      set_descriptors_dim_descriptors(descriptors, K*get_image_bands(image), error);
      break;}



    case 4:{ // METODO 4: SIFT, Kmeans (con descriptores SIFT) y VLAD
      kmeans_parameter_t params ;
      kmeans_model_t model_kmeans ;
      sift_model_t model_sift ;
      int H1, V1 ;
      int partial_sum = 0;
      unsigned int* raw_features;
      int iacum=0,n, x;
      int* parts;
      double* part_mean;

      start_crono( "SIFT" ) ;

      model_sift = sift_features ( image, get_segmentation_data(seg), command_arguments->sift_thresholds ) ;

      // preparacion de datos de sift para kmeans
      raw_features = (unsigned int*)malloc(model_sift.total_descriptors * dim_sift_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_sift.num_segments;i++){
        for(unsigned int j=0;j<model_sift.descriptors[i].size();j++){
          for(int k=0;k<dim_sift_descriptor;k++){
            raw_features[partial_sum*dim_sift_descriptor + k] = (int) model_sift.descriptors[i][j].desc[k];
          }
          partial_sum ++;
        }
      }

      stop_crono ( ) ;



      start_crono( "KMEANS" ) ;

      kmeans( raw_features , model_sift.total_descriptors, dim_sift_descriptor, params,  &model_kmeans ) ;

      stop_crono ( ) ;

      ////////// Opcion D - Media posteriori ////
      if(dim_sift_descriptor > get_image_bands(image)){
        x = dim_sift_descriptor;
        n = get_image_bands(image);
      }else{
        n = dim_sift_descriptor;
        x = get_image_bands(image);
      }
      parts = force_integer_splits(n, x);
      part_mean = (double *)calloc(model_kmeans.K*get_image_bands(image),sizeof(double));
      for(int i=0;i<model_kmeans.K;i++){
        iacum=0;
        for(int j=0;j<n;j++){
          for(int pi=0;pi<parts[j];pi++){
            part_mean[i*get_image_bands(image)+j] = part_mean[i*get_image_bands(image)+j] + model_kmeans.c[i*dim_sift_descriptor+iacum];
            iacum++;
          }
          part_mean[i*get_image_bands(image)+j] = part_mean[i*get_image_bands(image)+j] / parts[j];
        }
      }
      free(model_kmeans.c);
      model_kmeans.B=get_image_bands(image);
      model_kmeans.c=(double *)calloc(model_kmeans.K*model_kmeans.B,sizeof(double));
      for(int i=0;i<model_kmeans.K*model_kmeans.B;i++){
        model_kmeans.c[i] = part_mean[i];
      }
      free(part_mean);
      ////////////////////////////////////////////////////////////////

      ////////// Opcion E - PCA posteriori (solo para bandas < dim_sift) ////
      // Eigen::MatrixXf pca_data_matrix(model_kmeans.K, dim_sift_descriptor );
      // for(int i=0; i<model_kmeans.K; i++){
      // 	for(int j=0; j<dim_sift_descriptor; j++){
      // 		pca_data_matrix(i,j)=model_kmeans.c[i*dim_sift_descriptor+j];
      // 	}
      // }
      //
      // pca_t<float> pca;
      // pca.set_input(pca_data_matrix);
      // pca.compute();
      //
      // free(model_kmeans.c);
      // model_kmeans.B=get_image_bands(image);
      // model_kmeans.c=(double *)calloc(model_kmeans.K*model_kmeans.B,sizeof(double));
      // for(int k=0;k<model_kmeans.K;k++){
      // 	for(unsigned int i=0;i<get_image_bands(image);i++){
      //     model_kmeans.c[k*get_image_bands(image)+i]=0;
      //     for(int j=0;j<dim_sift_descriptor;j++){
      // 		    model_kmeans.c[k*get_image_bands(image)+i] = model_kmeans.c[k*get_image_bands(image)+i] + pca_data_matrix(k,j)*pca.get_eigen_vectors()(j,i);
      //     }
      // 	}
      // }
      ////////////////////////////////////////////////////////////////

      start_crono( "VLAD" ) ;

      data = vlad( get_image_data(image), get_segmentation_data(seg), get_image_bands(image), get_image_width(image), get_image_height(image), model_kmeans, H1 , V1, K) ;

      stop_crono ( ) ;

      destroy_kmeans_model( model_kmeans ) ;

      set_descriptors_dim_descriptors(descriptors, K*get_image_bands(image), error);
      break;}




    case 5:{ //METODO 5: SIFT, GMM y fisher vectors
      gmm_parameter_t gmm_params;
      sift_model_t model_sift ;
      gmm_model_t model_gmm;
      int partial_sum = 0;
      unsigned int * raw_features;
      int *parts = NULL, iacum=0, n, x;
      double* part_means=NULL, *part_covs=NULL;


      start_crono( "SIFT" ) ;

      model_sift = sift_features ( image, get_segmentation_data(seg), command_arguments->sift_thresholds ) ;

      // preparacion de datos de sift para kmeans
      raw_features = (unsigned int*)malloc(model_sift.total_descriptors * dim_sift_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_sift.num_segments;i++){
        for(unsigned int j=0;j<model_sift.descriptors[i].size();j++){
          for(int k=0;k<dim_sift_descriptor;k++){
            raw_features[partial_sum*dim_sift_descriptor + k] = (int) model_sift.descriptors[i][j].desc[k];
          }
          partial_sum ++;
        }
      }

      stop_crono ( ) ;



      start_crono( "GMM" ) ;
      gmm_params.dimensions = dim_sift_descriptor;
      gmm_params.numPixels = model_sift.total_descriptors;
      model_gmm = gmm ( raw_features, gmm_params ) ;

      stop_crono ( ) ;

      ////////// Opcion D - Media posteriori ////
      if(dim_sift_descriptor > get_image_bands(image)){
        x = dim_sift_descriptor;
        n = get_image_bands(image);
      }else{
        n = dim_sift_descriptor;
        x = get_image_bands(image);
      }
      parts = force_integer_splits(n, x);
      part_means = (double *)calloc(model_gmm.centers*get_image_bands(image),sizeof(double));
      part_covs = (double *)calloc(model_gmm.centers*get_image_bands(image),sizeof(double));
      for(int i=0;i<model_gmm.centers;i++){
        iacum=0;
        for(int j=0;j<n;j++){
          for(int pi=0;pi<parts[j];pi++){
            part_means[i*get_image_bands(image)+j] = part_means[i*get_image_bands(image)+j] + model_gmm.means[i*dim_sift_descriptor+iacum];
            part_covs[i*get_image_bands(image)+j] = part_covs[i*get_image_bands(image)+j] + model_gmm.covs[i*dim_sift_descriptor+iacum];
            iacum++;
          }
          part_means[i*get_image_bands(image)+j] = part_means[i*get_image_bands(image)+j] / parts[j];
          part_covs[i*get_image_bands(image)+j] = part_covs[i*get_image_bands(image)+j] / parts[j];
        }
      }
      free(model_gmm.means);
      free(model_gmm.covs);
      model_gmm.dimensions=get_image_bands(image);
      model_gmm.means=(double *)calloc(model_gmm.centers*model_gmm.dimensions,sizeof(double));
      model_gmm.covs=(double *)calloc(model_gmm.centers*model_gmm.dimensions,sizeof(double));
      for(int i=0;i<model_gmm.centers*model_gmm.dimensions;i++){
        model_gmm.means[i] = part_means[i];
        model_gmm.covs[i] = part_covs[i];
      }
      free(part_means);
      free(part_covs);
      ////////////////////////////////////////////////////////////////


      ////////// Opcion E - PCA posteriori (solo para bandas < dim_sift) ////
      // Eigen::MatrixXf pca_data_matrix_means(model_gmm.centers, dim_sift_descriptor );
      // Eigen::MatrixXf pca_data_matrix_covs(model_gmm.centers, dim_sift_descriptor );
      // for(int i=0; i<model_gmm.centers; i++){
      // 	for(int j=0; j<dim_sift_descriptor; j++){
      // 		pca_data_matrix_means(i,j)=model_gmm.means[i*dim_sift_descriptor+j];
      //     pca_data_matrix_covs(i,j)=model_gmm.covs[i*dim_sift_descriptor+j];
      // 	}
      // }
      //
      // pca_t<float> pca_means, pca_covs;
      // pca_means.set_input(pca_data_matrix_means);
      // pca_means.compute();
      // pca_covs.set_input(pca_data_matrix_covs);
      // pca_covs.compute();
      //
      // free(model_gmm.means);
      // free(model_gmm.covs);
      // model_gmm.dimensions=get_image_bands(image);
      // model_gmm.means=(double *)calloc(model_gmm.centers*model_gmm.dimensions,sizeof(double));
      // model_gmm.covs=(double *)calloc(model_gmm.centers*model_gmm.dimensions,sizeof(double));
      // for(int k=0;k<model_gmm.centers;k++){
      // 	for(unsigned int i=0;i<get_image_bands(image);i++){
      //     model_gmm.means[k*get_image_bands(image)+i]=0;
      //     model_gmm.covs[k*get_image_bands(image)+i]=0;
      //     for(int j=0;j<dim_sift_descriptor;j++){
      // 		    model_gmm.means[k*get_image_bands(image)+i] = model_gmm.means[k*get_image_bands(image)+i] + pca_data_matrix_means(k,j)*pca_means.get_eigen_vectors()(j,i);
      //         model_gmm.covs[k*get_image_bands(image)+i] = model_gmm.covs[k*get_image_bands(image)+i] + pca_data_matrix_covs(k,j)*pca_covs.get_eigen_vectors()(j,i);
      //     }
      // 	}
      // }
      ////////////////////////////////////////////////////////////////

      start_crono( "FisherVectors" ) ;

      data = fishervectors_features( image, get_segmentation_data(seg), model_gmm )  ;

      stop_crono ( ) ;

      set_descriptors_dim_descriptors(descriptors, K*get_image_bands(image), error);
      break;}




    case 6:{ // METODO 6: SIFT, Kmeans (con descriptores SIFT) y VLAD (con descriptores SIFT)
      kmeans_parameter_t params ;
      kmeans_model_t model_kmeans ;
      sift_model_t model_sift ;
      int H1 ;
      int partial_sum = 0;
      unsigned int* raw_features;

      start_crono( "SIFT" ) ;

      model_sift = sift_features ( image, get_segmentation_data(seg), command_arguments->sift_thresholds ) ;

      // preparacion de datos de sift para kmeans
      raw_features = (unsigned int*)malloc(model_sift.total_descriptors * dim_sift_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_sift.num_segments;i++){
        for(unsigned int j=0;j<model_sift.descriptors[i].size();j++){
          for(int k=0;k<dim_sift_descriptor;k++){
            raw_features[partial_sum*dim_sift_descriptor + k] = (int) model_sift.descriptors[i][j].desc[k];
          }
          partial_sum ++;
        }
      }

      stop_crono ( ) ;



      start_crono( "KMEANS" ) ;

      kmeans( raw_features , model_sift.total_descriptors, dim_sift_descriptor, params,  &model_kmeans ) ;

      stop_crono ( ) ;


      start_crono( "VLAD" ) ;

      data = vlad_sift( model_sift, dim_sift_descriptor, get_image_width(image), get_image_height(image), model_kmeans, H1 , K) ;

      stop_crono ( ) ;

      destroy_kmeans_model( model_kmeans ) ;

      set_descriptors_dim_descriptors(descriptors, K*get_image_bands(image), error);
      break;}




      case 7:{ // METODO 7: DSIFT, Kmeans (con descriptores SIFT) y VLAD
        kmeans_parameter_t params ;
        kmeans_model_t model_kmeans ;
        sift_model_t model_dsift ;
        int H1, V1 ;
        int partial_sum = 0;
        unsigned int* raw_features;
        int *parts = NULL, iacum, x, n;
        double* part_mean;

        start_crono( "DSIFT" ) ;

        model_dsift = dsift_features ( image, get_segmentation_data(seg), command_arguments->dsift_parameters) ;

        // preparacion de datos de sift para kmeans
        raw_features = (unsigned int*)malloc(model_dsift.total_descriptors * dim_sift_descriptor * sizeof(unsigned int));
        for(int i=0;i<model_dsift.num_segments;i++){
          for(unsigned int j=0;j<model_dsift.descriptors[i].size();j++){
            for(int k=0;k<dim_sift_descriptor;k++){
              raw_features[partial_sum*dim_sift_descriptor + k] = (int) model_dsift.descriptors[i][j].desc[k];
            }
            partial_sum ++;
          }
        }

        stop_crono ( ) ;



        start_crono( "KMEANS" ) ;

        kmeans( raw_features , model_dsift.total_descriptors, dim_sift_descriptor, params,  &model_kmeans ) ;

        stop_crono ( ) ;

        ////////// Opcion D - Media posteriori ////
        if(dim_sift_descriptor > get_image_bands(image)){
          x = dim_sift_descriptor;
          n = get_image_bands(image);
        }else{
          n = dim_sift_descriptor;
          x = get_image_bands(image);
        }
        parts = force_integer_splits(n, x);
        part_mean = (double *)calloc(model_kmeans.K*get_image_bands(image),sizeof(double));
        for(int i=0;i<model_kmeans.K;i++){
          iacum=0;
          for(int j=0;j<n;j++){
            for(int pi=0;pi<parts[j];pi++){
              part_mean[i*get_image_bands(image)+j] = part_mean[i*get_image_bands(image)+j] + model_kmeans.c[i*dim_sift_descriptor+iacum];
              iacum++;
            }
            part_mean[i*get_image_bands(image)+j] = part_mean[i*get_image_bands(image)+j] / parts[j];
          }
        }
        free(model_kmeans.c);
        model_kmeans.B=get_image_bands(image);
        model_kmeans.c=(double *)calloc(model_kmeans.K*model_kmeans.B,sizeof(double));
        for(int i=0;i<model_kmeans.K*model_kmeans.B;i++){
          model_kmeans.c[i] = part_mean[i];
        }
        free(part_mean);
        ////////////////////////////////////////////////////////////////

        ////////// Opcion E - PCA posteriori (solo para bandas < dim_sift) ////
        // Eigen::MatrixXf pca_data_matrix(model_kmeans.K, dim_sift_descriptor );
        // for(int i=0; i<model_kmeans.K; i++){
        // 	for(int j=0; j<dim_sift_descriptor; j++){
        // 		pca_data_matrix(i,j)=model_kmeans.c[i*dim_sift_descriptor+j];
        // 	}
        // }
        //
        // pca_t<float> pca;
        // pca.set_input(pca_data_matrix);
        // pca.compute();
        //
        // free(model_kmeans.c);
        // model_kmeans.B=get_image_bands(image);
        // model_kmeans.c=(double *)calloc(model_kmeans.K*model_kmeans.B,sizeof(double));
        // for(int k=0;k<model_kmeans.K;k++){
        // 	for(unsigned int i=0;i<get_image_bands(image);i++){
        //     model_kmeans.c[k*get_image_bands(image)+i]=0;
        //     for(int j=0;j<dim_sift_descriptor;j++){
        // 		    model_kmeans.c[k*get_image_bands(image)+i] = model_kmeans.c[k*get_image_bands(image)+i] + pca_data_matrix(k,j)*pca.get_eigen_vectors()(j,i);
        //     }
        // 	}
        // }
        ////////////////////////////////////////////////////////////////
        
        start_crono( "VLAD" ) ;

        data = vlad( get_image_data(image), get_segmentation_data(seg), get_image_bands(image), get_image_width(image), get_image_height(image), model_kmeans, H1 , V1, K) ;

        stop_crono ( ) ;

        destroy_kmeans_model( model_kmeans ) ;

        set_descriptors_dim_descriptors(descriptors, K*get_image_bands(image), error);
        break;}




      case 8:{ //METODO 8: DSIFT, GMM y fisher vectors
        gmm_parameter_t gmm_params;
        sift_model_t model_dsift ;
        gmm_model_t model_gmm;
        int partial_sum = 0;
        unsigned int * raw_features;
        int *parts = NULL, iacum=0, x, n;
        double* part_means=NULL, *part_covs=NULL;


        start_crono( "DSIFT" ) ;

        model_dsift = dsift_features ( image, get_segmentation_data(seg), command_arguments->dsift_parameters ) ;

        // preparacion de datos de dsift para kmeans
        raw_features = (unsigned int*)malloc(model_dsift.total_descriptors * dim_sift_descriptor * sizeof(unsigned int));
        for(int i=0;i<model_dsift.num_segments;i++){
          for(unsigned int j=0;j<model_dsift.descriptors[i].size();j++){
            for(int k=0;k<dim_sift_descriptor;k++){
              raw_features[partial_sum*dim_sift_descriptor + k] = (int) model_dsift.descriptors[i][j].desc[k];
            }
            partial_sum ++;
          }
        }

        stop_crono ( ) ;



        start_crono( "GMM" ) ;
        gmm_params.dimensions = dim_sift_descriptor;
        gmm_params.numPixels = model_dsift.total_descriptors;
        model_gmm = gmm ( raw_features, gmm_params ) ;

        stop_crono ( ) ;

        ////////// Opcion D - Media posteriori ////
        if(dim_sift_descriptor > get_image_bands(image)){
          x = dim_sift_descriptor;
          n = get_image_bands(image);
        }else{
          n = dim_sift_descriptor;
          x = get_image_bands(image);
        }
        parts = force_integer_splits(n, x);
        part_means = (double *)calloc(model_gmm.centers*get_image_bands(image),sizeof(double));
        part_covs = (double *)calloc(model_gmm.centers*get_image_bands(image),sizeof(double));
        for(int i=0;i<model_gmm.centers;i++){
          iacum=0;
          for(int j=0;j<n;j++){
            for(int pi=0;pi<parts[j];pi++){
              part_means[i*get_image_bands(image)+j] = part_means[i*get_image_bands(image)+j] + model_gmm.means[i*dim_sift_descriptor+iacum];
              part_covs[i*get_image_bands(image)+j] = part_covs[i*get_image_bands(image)+j] + model_gmm.covs[i*dim_sift_descriptor+iacum];
              iacum++;
            }
            part_means[i*get_image_bands(image)+j] = part_means[i*get_image_bands(image)+j] / parts[j];
            part_covs[i*get_image_bands(image)+j] = part_covs[i*get_image_bands(image)+j] / parts[j];
          }
        }
        free(model_gmm.means);
        free(model_gmm.covs);
        model_gmm.dimensions=get_image_bands(image);
        model_gmm.means=(double *)calloc(model_gmm.centers*model_gmm.dimensions,sizeof(double));
        model_gmm.covs=(double *)calloc(model_gmm.centers*model_gmm.dimensions,sizeof(double));
        for(int i=0;i<model_gmm.centers*model_gmm.dimensions;i++){
          model_gmm.means[i] = part_means[i];
          model_gmm.covs[i] = part_covs[i];
        }
        free(part_means);
        free(part_covs);
        ////////////////////////////////////////////////////////////////


        ////////// Opcion E - PCA posteriori (solo para bandas < dim_sift) ////
        // Eigen::MatrixXf pca_data_matrix_means(model_gmm.centers, dim_sift_descriptor );
        // Eigen::MatrixXf pca_data_matrix_covs(model_gmm.centers, dim_sift_descriptor );
        // for(int i=0; i<model_gmm.centers; i++){
        // 	for(int j=0; j<dim_sift_descriptor; j++){
        // 		pca_data_matrix_means(i,j)=model_gmm.means[i*dim_sift_descriptor+j];
        //     pca_data_matrix_covs(i,j)=model_gmm.covs[i*dim_sift_descriptor+j];
        // 	}
        // }
        //
        // pca_t<float> pca_means, pca_covs;
        // pca_means.set_input(pca_data_matrix_means);
        // pca_means.compute();
        // pca_covs.set_input(pca_data_matrix_covs);
        // pca_covs.compute();
        //
        // free(model_gmm.means);
        // free(model_gmm.covs);
        // model_gmm.dimensions=get_image_bands(image);
        // model_gmm.means=(double *)calloc(model_gmm.centers*model_gmm.dimensions,sizeof(double));
        // model_gmm.covs=(double *)calloc(model_gmm.centers*model_gmm.dimensions,sizeof(double));
        // for(int k=0;k<model_gmm.centers;k++){
        // 	for(unsigned int i=0;i<get_image_bands(image);i++){
        //     model_gmm.means[k*get_image_bands(image)+i]=0;
        //     model_gmm.covs[k*get_image_bands(image)+i]=0;
        //     for(int j=0;j<dim_sift_descriptor;j++){
        // 		    model_gmm.means[k*get_image_bands(image)+i] = model_gmm.means[k*get_image_bands(image)+i] + pca_data_matrix_means(k,j)*pca_means.get_eigen_vectors()(j,i);
        //         model_gmm.covs[k*get_image_bands(image)+i] = model_gmm.covs[k*get_image_bands(image)+i] + pca_data_matrix_covs(k,j)*pca_covs.get_eigen_vectors()(j,i);
        //     }
        // 	}
        // }
        ////////////////////////////////////////////////////////////////

        start_crono( "FisherVectors" ) ;

        data = fishervectors_features( image, get_segmentation_data(seg), model_gmm )  ;

        stop_crono ( ) ;

        set_descriptors_dim_descriptors(descriptors, K*get_image_bands(image), error);
        break;}




      case 9:{ // METODO 9: DSIFT, Kmeans (con descriptores DSIFT) y VLAD (con descriptores SIFT)
        kmeans_parameter_t params ;
        kmeans_model_t model_kmeans ;
        sift_model_t model_dsift ;
        int H1 ;
        int partial_sum = 0;
        unsigned int* raw_features;

        start_crono( "DSIFT" ) ;

        model_dsift = dsift_features ( image, get_segmentation_data(seg), command_arguments->dsift_parameters ) ;

        // preparacion de datos de dsift para kmeans
        raw_features = (unsigned int*)malloc(model_dsift.total_descriptors * dim_sift_descriptor * sizeof(unsigned int));
        for(int i=0;i<model_dsift.num_segments;i++){
          for(unsigned int j=0;j<model_dsift.descriptors[i].size();j++){
            for(int k=0;k<dim_sift_descriptor;k++){
              raw_features[partial_sum*dim_sift_descriptor + k] = (int) model_dsift.descriptors[i][j].desc[k];
            }
            partial_sum ++;
          }
        }

        stop_crono ( ) ;



        start_crono( "KMEANS" ) ;

        kmeans( raw_features , model_dsift.total_descriptors, dim_sift_descriptor, params,  &model_kmeans ) ;

        stop_crono ( ) ;


        start_crono( "VLAD" ) ;

        data = vlad_sift( model_dsift, dim_sift_descriptor, get_image_width(image), get_image_height(image), model_kmeans, H1 , K) ;

        stop_crono ( ) ;

        destroy_kmeans_model( model_kmeans ) ;

        set_descriptors_dim_descriptors(descriptors, K*get_image_bands(image), error);
        break;}




    default:{
      print_error((char*)"Texture method not recognized");
      exit(EXIT_FAILURE);}

  }



  set_descriptors_number_descriptors(descriptors, get_segmentation_number_segments(seg), error);
  set_descriptors_data(descriptors, data, error);
  free(data);
  labels = get_labels_per_segment(seg, gt_train, get_descriptors_number_descriptors(descriptors));
  set_descriptors_labels(descriptors, labels, error);
  free(labels);

  return descriptors;
}
