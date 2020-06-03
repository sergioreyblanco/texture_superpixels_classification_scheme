
/**
			  * @file				texture_pipelines.cpp
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Different grouping of texture algorithm for texture descriptors obtaining.
			  */

#include "texture_pipelines.h"


void reduce_dim_after_clustering(image_struct* image, kmeans_model_t *model_kmeans, gmm_model_t *model_gmm, int dim_desc, int type){
  if((unsigned int) dim_desc < get_image_bands(image)){
    print_error((char *)"Error in Bands/ Dim Descriptor dimension");
    exit(EXIT_FAILURE);
  }

  switch(type){



    case 1:{ //media posteriori kmeans
      int iacum=0, x=0, n=0;
      int* parts;
      double* part_mean;

      x = dim_desc;
      n = get_image_bands(image);

      parts = force_integer_splits(n, x);
      part_mean = (double *)calloc(model_kmeans->K*get_image_bands(image),sizeof(double));
      for(int i=0;i<model_kmeans->K;i++){
        iacum=0;
        for(int j=0;j<n;j++){
          for(int pi=0;pi<parts[j];pi++){
            part_mean[i*get_image_bands(image)+j] = part_mean[i*get_image_bands(image)+j] + model_kmeans->c[i*dim_desc+iacum];
            iacum++;
          }
          part_mean[i*get_image_bands(image)+j] = part_mean[i*get_image_bands(image)+j] / parts[j];
        }
      }
      free(model_kmeans->c);
      model_kmeans->B=get_image_bands(image);
      model_kmeans->c=(double *)calloc(model_kmeans->K*model_kmeans->B,sizeof(double));
      for(int i=0;i<model_kmeans->K*model_kmeans->B;i++){
        model_kmeans->c[i] = part_mean[i];
      }
      free(part_mean);

    break;}



    case 2:{ //media posteriori GMM
      int iacum=0, x=0, n=0;
      int* parts;
      double* part_means, *part_covs;

      x = dim_desc;
      n = get_image_bands(image);

      parts = force_integer_splits(n, x);
      part_means = (double *)calloc(model_gmm->centers*get_image_bands(image),sizeof(double));
      part_covs = (double *)calloc(model_gmm->centers*get_image_bands(image),sizeof(double));
      for(int i=0;i<model_gmm->centers;i++){
        iacum=0;
        for(int j=0;j<n;j++){
          for(int pi=0;pi<parts[j];pi++){
            part_means[i*get_image_bands(image)+j] = part_means[i*get_image_bands(image)+j] + model_gmm->means[i*dim_desc+iacum];
            part_covs[i*get_image_bands(image)+j] = part_covs[i*get_image_bands(image)+j] + model_gmm->covs[i*dim_desc+iacum];
            iacum++;
          }
          part_means[i*get_image_bands(image)+j] = part_means[i*get_image_bands(image)+j] / parts[j];
          part_covs[i*get_image_bands(image)+j] = part_covs[i*get_image_bands(image)+j] / parts[j];
        }
      }
      free(model_gmm->means);
      free(model_gmm->covs);
      model_gmm->dimensions=get_image_bands(image);
      model_gmm->means=(double *)calloc(model_gmm->centers*model_gmm->dimensions,sizeof(double));
      model_gmm->covs=(double *)calloc(model_gmm->centers*model_gmm->dimensions,sizeof(double));
      for(int i=0;i<model_gmm->centers*model_gmm->dimensions;i++){
        model_gmm->means[i] = part_means[i];
        model_gmm->covs[i] = part_covs[i];
      }
      free(part_means);
      free(part_covs);
    break;}



    case 3:{ //PCA posteriori kmeans

      Eigen::MatrixXf pca_data_matrix(model_kmeans->K, dim_desc );
      for(int i=0; i<model_kmeans->K; i++){
      	for(int j=0; j<dim_desc; j++){
      		pca_data_matrix(i,j)=model_kmeans->c[i*dim_desc+j];
      	}
      }

      pca_t<float> pca;
      pca.set_input(pca_data_matrix);
      pca.compute();

      free(model_kmeans->c);
      model_kmeans->B=get_image_bands(image);
      model_kmeans->c=(double *)calloc(model_kmeans->K*model_kmeans->B,sizeof(double));
      for(int k=0;k<model_kmeans->K;k++){
      	for(unsigned int i=0;i<get_image_bands(image);i++){
          model_kmeans->c[k*get_image_bands(image)+i]=0;
          for(int j=0;j<dim_desc;j++){
      		    model_kmeans->c[k*get_image_bands(image)+i] = model_kmeans->c[k*get_image_bands(image)+i] + pca_data_matrix(k,j)*pca.get_eigen_vectors()(j,i);
          }
      	}
      }

    break;}



    case 6:{ //PCA posteriori GMM

      Eigen::MatrixXf pca_data_matrix_means(model_gmm->centers, dim_desc );
      Eigen::MatrixXf pca_data_matrix_covs(model_gmm->centers, dim_desc );
      for(int i=0; i<model_gmm->centers; i++){
      	for(int j=0; j<dim_desc; j++){
      		pca_data_matrix_means(i,j)=model_gmm->means[i*dim_desc+j];
          pca_data_matrix_covs(i,j)=model_gmm->covs[i*dim_desc+j];
      	}
      }

      pca_t<float> pca_means, pca_covs;
      pca_means.set_input(pca_data_matrix_means);
      pca_means.compute();
      pca_covs.set_input(pca_data_matrix_covs);
      pca_covs.compute();

      free(model_gmm->means);
      free(model_gmm->covs);
      model_gmm->dimensions=get_image_bands(image);
      model_gmm->means=(double *)calloc(model_gmm->centers*model_gmm->dimensions,sizeof(double));
      model_gmm->covs=(double *)calloc(model_gmm->centers*model_gmm->dimensions,sizeof(double));
      for(int k=0;k<model_gmm->centers;k++){
      	for(unsigned int i=0;i<get_image_bands(image);i++){
          model_gmm->means[k*get_image_bands(image)+i]=0;
          model_gmm->covs[k*get_image_bands(image)+i]=0;
          for(int j=0;j<dim_desc;j++){
      		    model_gmm->means[k*get_image_bands(image)+i] = model_gmm->means[k*get_image_bands(image)+i] + pca_data_matrix_means(k,j)*pca_means.get_eigen_vectors()(j,i);
              model_gmm->covs[k*get_image_bands(image)+i] = model_gmm->covs[k*get_image_bands(image)+i] + pca_data_matrix_covs(k,j)*pca_covs.get_eigen_vectors()(j,i);
          }
      	}
      }
    break;}
    }

}



void reduce_dim_before_descriptors(image_struct* image, image_struct* image_aux, int dim_desc, int type, char* error){

  int * part_mean=NULL;

  set_image_path(image_aux, "", error);
  set_image_size(image_aux, (size_t) 0, error);
  set_image_width(image_aux, get_image_width(image), error);
  set_image_height(image_aux, get_image_height(image), error);
  set_image_bands(image_aux, dim_desc, error);


  switch(type){

    case 1:{ // Media
      int iacum=0, x=0, n=0;
      int* parts;

      x = get_image_bands(image);
      n = dim_desc;

      parts = force_integer_splits(n, x);
      part_mean = (int *)calloc(get_image_width(image_aux)*get_image_height(image_aux)*dim_desc,sizeof(int));
      for(unsigned int i=0;i<get_image_width(image_aux)*get_image_height(image_aux);i++){
        iacum=0;
        for(int j=0;j<n;j++){
          for(int pi=0;pi<parts[j];pi++){
            part_mean[i*dim_desc+j] = part_mean[i*dim_desc+j] + get_image_data(image)[i*get_image_bands(image)+iacum];
            iacum++;
          }
          part_mean[i*dim_desc+j] = part_mean[i*dim_desc+j] / parts[j];
        }
      }
    break;}

    case 3:{ // PCA
      Eigen::MatrixXf pca_data_matrix(get_image_width(image_aux)*get_image_height(image_aux), get_image_bands(image) );
      for(unsigned int i=0; i<get_image_width(image_aux)*get_image_height(image_aux); i++){
      	for(int j=0; j<dim_desc; j++){
      		pca_data_matrix(i,j)=get_image_data(image)[i*get_image_bands(image)+j];
      	}
      }

      pca_t<float> pca;
      pca.set_input(pca_data_matrix);
      pca.compute();

      part_mean=(int *)calloc(get_image_width(image_aux)*get_image_height(image_aux)*dim_desc,sizeof(int));
      for(unsigned int k=0;k<get_image_width(image_aux)*get_image_height(image_aux);k++){
      	for( int i=0;i<dim_desc;i++){
          part_mean[k*dim_desc+i]=0;
          for(unsigned int j=0;j<get_image_bands(image);j++){
      		    part_mean[k*dim_desc+i] = part_mean[k*dim_desc+i] + pca_data_matrix(k,j)*pca.get_eigen_vectors()(j,i);
          }
      	}
      }

    break;}

  }

  set_image_data(image_aux, part_mean, error);
}




texture_struct* texture_pipeline(image_struct* image, image_struct* train_image,  segmentation_struct* seg, reference_data_struct* gt_train, int num_pixels, command_arguments_struct *command_arguments, char* error){
  texture_struct * descriptors = (texture_struct*)malloc(sizeof(texture_struct));
  unsigned int dim_sift_descriptor=128;
  double* data, *data_std;
  int *labels;
  int K=32, dim=0;


  switch(get_command_arguments_texture_pipeline(command_arguments)){





    case 0:{ // METODO 0: sin texturas (media de cada segmento)

      //start_crono( "MEANS COMPUTATION" ) ;
      start_crono( "CENTRAL POINT COMPUTATION" ) ;

      //data = get_means_per_segment(seg, image, get_segmentation_number_segments(seg));
      data = get_central_pixel_per_segment(seg, image, get_segmentation_number_segments(seg));
      dim = get_image_bands(image);

      stop_crono ( ) ;

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
      dim = K*get_image_bands(image);
      destroy_kmeans_model( model ) ;

      stop_crono ( ) ;

      break;}





    case 2:{ // METODO 2: Kmeans y BOW

      kmeans_parameter_t params ;
      kmeans_model_t model ;
      int H1, V1 ;

      start_crono( "KMEANS" ) ;

      kmeans( get_image_data(image) , num_pixels, get_image_bands(image), params,  &model ) ;
      K=model.K;

      stop_crono ( ) ;



      start_crono( "BOW" ) ;

      data = bow( get_image_data(image), get_segmentation_data(seg), get_image_bands(image), get_image_width(image), get_image_height(image), model, H1 , V1, K ) ;
      dim = K*get_image_bands(image);
      destroy_kmeans_model( model ) ;

      stop_crono ( ) ;

      break;}





    case 3:{ //METODO 3: GMM y fisher vectors

      gmm_parameter_t gmm_params;

      start_crono( "GMM" ) ;
      gmm_params.dimensions = get_image_bands(image);
      gmm_params.numPixels = num_pixels;
      printf("%d\n", num_pixels);
      gmm_model_t model = gmm ( get_image_data(image), gmm_params ) ;

      stop_crono ( ) ;



      start_crono( "FisherVectors" ) ;

      data = fishervectors_features( get_image_data(image), seg, model )  ;
      dim = K*get_image_bands(image);

      stop_crono ( ) ;

      break;}





    case 4:{ // METODO 4: SIFT, Kmeans (con descriptores SIFT) y VLAD
      kmeans_parameter_t params ;
      kmeans_model_t model_kmeans ;
      descriptor_model_t model_sift ;
      int H1, V1, partial_sum = 0;
      unsigned int * raw_features;
      image_struct* image_aux = (image_struct*)malloc(sizeof(image_struct));



      start_crono( "SIFT" ) ;

      if(dim_sift_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_sift_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_sift = sift_features ( image_aux, get_segmentation_data(seg), (float*)get_command_arguments_sift_thresholds(command_arguments) ) ;
      } else{
        model_sift = sift_features ( image, get_segmentation_data(seg), (float*)get_command_arguments_sift_thresholds(command_arguments) ) ;
      }

      // preparacion de datos de sift para kmeans
      raw_features = (unsigned int*)malloc(model_sift.total_descriptors * dim_sift_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_sift.num_segments;i++){
        for(unsigned int j=0;j<model_sift.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_sift_descriptor;k++){
            raw_features[partial_sum*dim_sift_descriptor + k] = (int) model_sift.descriptors[i][j].desc[k];
            //printf("%d  ", raw_features[partial_sum*dim_sift_descriptor + k]);
          }//printf("\n");
          partial_sum ++;
        }
      }

      stop_crono ( ) ;




      start_crono( "KMEANS" ) ;

      kmeans( raw_features , model_sift.total_descriptors, dim_sift_descriptor, params,  &model_kmeans ) ;

      stop_crono ( ) ;




      start_crono( "VLAD" ) ;

      if(dim_sift_descriptor > get_image_bands(image)){
        reduce_dim_after_clustering(image, &model_kmeans, NULL, dim_sift_descriptor, get_command_arguments_reduction_method(command_arguments));
        data = vlad( get_image_data(image), get_segmentation_data(seg), get_image_bands(image), get_image_width(image), get_image_height(image), model_kmeans, H1 , V1, K) ;
        dim = K*get_image_bands(image);
      } else{
        data = vlad( get_image_data(image_aux), get_segmentation_data(seg), dim_sift_descriptor, get_image_width(image_aux), get_image_height(image_aux), model_kmeans, H1 , V1, K) ;
        dim = K*get_image_bands(image_aux);
      }

      destroy_kmeans_model( model_kmeans ) ;

      stop_crono ( ) ;

      break;}





    case 5:{ //METODO 5: SIFT, GMM y fisher vectors
      gmm_parameter_t gmm_params;
      descriptor_model_t model_sift ;
      gmm_model_t model_gmm;
      int partial_sum = 0;
      unsigned int * raw_features;
      image_struct* image_aux = (image_struct*)malloc(sizeof(image_struct));


      start_crono( "SIFT" ) ;

      if(dim_sift_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_sift_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_sift = sift_features ( image_aux, get_segmentation_data(seg), (float*)get_command_arguments_sift_thresholds(command_arguments) ) ;
      } else{
        model_sift = sift_features ( image, get_segmentation_data(seg), (float*)get_command_arguments_sift_thresholds(command_arguments) ) ;
      }

      // preparacion de datos de sift para kmeans
      raw_features = (unsigned int*)malloc(model_sift.total_descriptors * dim_sift_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_sift.num_segments;i++){
        for(unsigned int j=0;j<model_sift.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_sift_descriptor;k++){
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



      start_crono( "FisherVectors" ) ;

      if(dim_sift_descriptor > get_image_bands(image)){
        reduce_dim_after_clustering(image, NULL, &model_gmm, dim_sift_descriptor, get_command_arguments_reduction_method(command_arguments)*2);
        data = fishervectors_features( get_image_data(image), seg, model_gmm )  ;
        dim = K*get_image_bands(image);
      } else{
        data = fishervectors_features( get_image_data(image_aux), seg, model_gmm )  ;
        dim = K*get_image_bands(image_aux);
      }

      stop_crono ( ) ;


      break;}





    case 6:{ // METODO 6: SIFT, Kmeans (con descriptores SIFT) y VLAD (descrs)
      kmeans_parameter_t params ;
      kmeans_model_t model_kmeans ;
      descriptor_model_t model_sift ;
      int H1 ;
      int partial_sum = 0;
      unsigned int* raw_features;
      int R4=5;     //reducir los descriptores finales de vlad
      image_struct* image_aux=(image_struct*)malloc(sizeof(image_struct));

      start_crono( "SIFT" ) ;

      if(dim_sift_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_sift_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_sift = sift_features ( image_aux, get_segmentation_data(seg), (float*)get_command_arguments_sift_thresholds(command_arguments) ) ;
      } else{
        model_sift = sift_features ( image, get_segmentation_data(seg), (float*)get_command_arguments_sift_thresholds(command_arguments) ) ;
      }


      //R4
      raw_features = (unsigned int*)malloc(model_sift.total_descriptors * dim_sift_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_sift.num_segments;i++){
        for(unsigned int j=0;j<model_sift.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_sift_descriptor;k++){
            raw_features[partial_sum*dim_sift_descriptor + k] = (int) model_sift.descriptors[i][j].desc[k];
          }
          partial_sum ++;
        }
      }


      stop_crono ( ) ;



      start_crono( "KMEANS" ) ;

      //R4
      kmeans( raw_features , model_sift.total_descriptors, dim_sift_descriptor, params,  &model_kmeans ) ;

      stop_crono ( ) ;


      start_crono( "VLAD" ) ;

      //R4 PCA
      double * data_aux = vlad_sift( model_sift, dim_sift_descriptor, get_image_width(image), get_image_height(image), model_kmeans, H1 , K) ;


      Eigen::MatrixXf pca_data_matrix(get_segmentation_number_segments(seg), model_kmeans.K * dim_sift_descriptor );
      for(unsigned int i=0; i<get_segmentation_number_segments(seg); i++){
      	for(unsigned int j=0; j<dim_sift_descriptor*model_kmeans.K; j++){
      		pca_data_matrix(i,j) = data_aux[i*dim_sift_descriptor+j];
      	}
      }

      pca_t<float> pca;
      pca.set_input(pca_data_matrix);
      pca.compute();

      double* part_vlad = (double *)calloc(get_segmentation_number_segments(seg)*R4,sizeof(double));
      for(unsigned int k=0;k<get_segmentation_number_segments(seg);k++){
      	for( int i=0;i<R4;i++){
          part_vlad[k*R4+i]=0;
          for(unsigned int j=0;j<dim_sift_descriptor;j++){
      		   part_vlad[k*R4+i] = part_vlad[k*R4+i] + pca_data_matrix(k,j)*pca.get_eigen_vectors()(j,i);
          }
      	}
      }

      // //R4 Media
      // double * data_aux = vlad_sift( model_sift, dim_sift_descriptor, get_image_width(image), get_image_height(image), model_kmeans, H1 , K) ;;
      //
      // int n=R4, x=K*dim_sift_descriptor;
      // int* parts =force_integer_splits(n, x);
      // int iacum;
      // double* part_vlad = (double *)calloc(H1*n,sizeof(double));
      // for(int i=0;i<H1;i++){
      //   iacum=0;
      //   for(int j=0;j<n;j++){
      //     for(int pi=0;pi<parts[j];pi++){
      //       part_vlad[i*n+j] = part_vlad[i*n+j] + data_aux[i*dim_sift_descriptor*model_kmeans.K+iacum];
      //       iacum++;
      //     }
      //     part_vlad[i*n+j] = part_vlad[i*n+j] / parts[j];
      //   }
      // }

      //Concatenando pixel al descriptor vlad
      //double* central_per_segment = get_central_pixel_per_segment(seg, image, get_segmentation_number_segments(seg));
      int* central_per_segment = get_means_per_segment(seg, image, get_segmentation_number_segments(seg));
      data = (double*) malloc(get_segmentation_number_segments(seg) * (R4+get_image_bands(image)) * sizeof(double));
      for(unsigned int i=0; i < get_segmentation_number_segments(seg); i++){
        for(unsigned int j=0; j < (R4+get_image_bands(image)); j++){
          if(j < (unsigned int)(R4)){
            data[i*(R4+get_image_bands(image))+j] = part_vlad[i*(R4)+j];
          }else{
            data[i*(R4+get_image_bands(image))+j] = central_per_segment[i*get_image_bands(image) + (j-(R4)) ];
          }
        }
      }

      dim = R4 + get_image_bands(image);



      destroy_kmeans_model( model_kmeans ) ;

      stop_crono ( ) ;


      break;}





    case 7:{ //METODO 7: SIFT, GMM y fisher vectors (descrs)
      gmm_parameter_t gmm_params;
      descriptor_model_t model_sift ;
      gmm_model_t model_gmm;
      int partial_sum = 0;
      unsigned int * raw_features;
      image_struct* image_aux = (image_struct*)malloc(sizeof(image_struct));
      int R4=5, d=0;


      start_crono( "SIFT" ) ;

      if(dim_sift_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_sift_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_sift = sift_features ( image_aux, get_segmentation_data(seg), (float*)get_command_arguments_sift_thresholds(command_arguments) ) ;
      } else{
        model_sift = sift_features ( image, get_segmentation_data(seg), (float*)get_command_arguments_sift_thresholds(command_arguments) ) ;
      }

      // preparacion de datos de sift para kmeans
      raw_features = (unsigned int*)malloc(model_sift.total_descriptors * dim_sift_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_sift.num_segments;i++){
        for(unsigned int j=0;j<model_sift.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_sift_descriptor;k++){
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



      start_crono( "FisherVectors" ) ;

      double* data_aux=NULL;
      if(dim_sift_descriptor > get_image_bands(image)){
        reduce_dim_after_clustering(image, NULL, &model_gmm, dim_sift_descriptor, get_command_arguments_reduction_method(command_arguments)*2);
        data_aux = fishervectors_features( get_image_data(image), seg, model_gmm )  ;
	      d=model_gmm.dimensions;
      } else{
        data_aux = fishervectors_features( get_image_data(image_aux), seg, model_gmm )  ;
	      d=dim_sift_descriptor;
      }


      //R4 PCA
      Eigen::MatrixXf pca_data_matrix(get_segmentation_number_segments(seg), model_gmm.centers * d );
      for(unsigned int i=0; i<get_segmentation_number_segments(seg); i++){
      	for(int j=0; j<d*model_gmm.centers; j++){
      		pca_data_matrix(i,j) = data_aux[i*d+j];
      	}
      }

      pca_t<float> pca;
      pca.set_input(pca_data_matrix);
      pca.compute();

      double* part_fisher = (double *)calloc(get_segmentation_number_segments(seg)*R4,sizeof(double));
      for(unsigned int k=0;k<get_segmentation_number_segments(seg);k++){
       	for( int i=0;i<R4;i++){
          part_fisher[k*R4+i]=0;
          for(int j=0;j<d;j++){
      		    part_fisher[k*R4+i] = part_fisher[k*R4+i] + pca_data_matrix(k,j)*pca.get_eigen_vectors()(j,i);
          }
      	}
      }


      // //R4 Media corregir gmdimensions
      //  int n=R4, x=model_gmm.centers*model_gmm.dimensions;
      //  int* parts =force_integer_splits(n, x);
      //  int iacum;
      //  double* part_fisher = (double *)calloc(get_segmentation_number_segments(seg)*n,sizeof(double));
      //
      //  for(unsigned int i=0;i<get_segmentation_number_segments(seg);i++){
      //    iacum=0;
      //    for(int j=0;j<n;j++){
      //      for(int pi=0;pi<parts[j];pi++){
      //        part_fisher[i*n+j] = part_fisher[i*n+j] + data_aux[i*model_gmm.dimensions*model_gmm.centers+iacum];
      //        iacum++;
      //      }
      //      part_fisher[i*n+j] = part_fisher[i*n+j] / parts[j];
      //    }
      //  }


      //Concatenando pixel al descriptor vlad
      //double* central_per_segment = get_central_pixel_per_segment(seg, image, get_segmentation_number_segments(seg));
      int* central_per_segment = get_means_per_segment(seg, image, get_segmentation_number_segments(seg));
      data = (double*) malloc(get_segmentation_number_segments(seg) * (R4+get_image_bands(image)) * sizeof(double));
      for(unsigned int i=0; i < get_segmentation_number_segments(seg); i++){
        for(unsigned int j=0; j < (R4+get_image_bands(image)); j++){
          if(j < (unsigned int)(R4)){
            data[i*(R4+get_image_bands(image))+j] = part_fisher[i*(R4)+j];
          }else{
            data[i*(R4+get_image_bands(image))+j] = central_per_segment[i*get_image_bands(image) + (j-(R4)) ];
          }
        }
      }

      stop_crono ( ) ;

      dim = R4 + get_image_bands(image);

      break;}





    case 8:{ // METODO 8: DSIFT, Kmeans (con descriptores SIFT) y VLAD
      kmeans_parameter_t params ;
      kmeans_model_t model_kmeans ;
      descriptor_model_t model_dsift ;
      int H1, V1 ;
      int partial_sum = 0;
      unsigned int* raw_features;
      image_struct* image_aux = (image_struct*)malloc(sizeof(image_struct));


      start_crono( "DSIFT" ) ;

      if(dim_sift_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_sift_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_dsift = dsift_features ( image_aux, get_segmentation_data(seg), (int*)get_command_arguments_dsift_parameters(command_arguments) ) ;
      } else{
        model_dsift = dsift_features ( image, get_segmentation_data(seg), (int*)get_command_arguments_dsift_parameters(command_arguments) ) ;
      }

      // preparacion de datos de sift para kmeans
      raw_features = (unsigned int*)malloc(model_dsift.total_descriptors * dim_sift_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_dsift.num_segments;i++){
        for(unsigned int j=0;j<model_dsift.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_sift_descriptor;k++){
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

      if(dim_sift_descriptor > get_image_bands(image)){
        reduce_dim_after_clustering(image, &model_kmeans, NULL, dim_sift_descriptor, get_command_arguments_reduction_method(command_arguments));
        data = vlad( get_image_data(image), get_segmentation_data(seg), get_image_bands(image), get_image_width(image), get_image_height(image), model_kmeans, H1 , V1, K) ;
        dim = K*get_image_bands(image);
      } else{
        data = vlad( get_image_data(image_aux), get_segmentation_data(seg), dim_sift_descriptor, get_image_width(image), get_image_height(image), model_kmeans, H1 , V1, K) ;
        dim = K*get_image_bands(image_aux);
      }
      destroy_kmeans_model( model_kmeans ) ;

      stop_crono ( ) ;


      break;}





    case 9:{ //METODO 9: DSIFT, GMM y fisher vectors
      gmm_parameter_t gmm_params;
      descriptor_model_t model_dsift ;
      gmm_model_t model_gmm;
      int partial_sum = 0;
      unsigned int * raw_features;
      image_struct* image_aux = (image_struct*)malloc(sizeof(image_struct));
      dim_sift_descriptor=128;


      start_crono( "DSIFT" ) ;

      if(dim_sift_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_sift_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_dsift = dsift_features ( image_aux, get_segmentation_data(seg), (int*)get_command_arguments_dsift_parameters(command_arguments) ) ;
      } else{
        model_dsift = dsift_features ( image, get_segmentation_data(seg), (int*)get_command_arguments_dsift_parameters(command_arguments) ) ;
      }

      // preparacion de datos de dsift para kmeans
      raw_features = (unsigned int*)malloc(model_dsift.total_descriptors * dim_sift_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_dsift.num_segments;i++){
        for(unsigned int j=0;j<model_dsift.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_sift_descriptor;k++){
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




      start_crono( "FisherVectors" ) ;

      if(dim_sift_descriptor > get_image_bands(image)){
        reduce_dim_after_clustering(image, NULL, &model_gmm, dim_sift_descriptor, get_command_arguments_reduction_method(command_arguments)*2);
        data = fishervectors_features( get_image_data(image), seg, model_gmm )  ;
        dim = K*get_image_bands(image);
      } else{
        data = fishervectors_features( get_image_data(image_aux), seg, model_gmm )  ;
        dim = K*get_image_bands(image_aux);
      }

      stop_crono ( ) ;


      break;}





    case 10:{ // METODO 10: DSIFT, Kmeans (con descriptores DSIFT) y VLAD (descrs)
      kmeans_parameter_t params ;
      kmeans_model_t model_kmeans ;
      descriptor_model_t model_dsift ;
      int H1 ;
      int partial_sum = 0;
      unsigned int* raw_features;
      int R4=5;     //reducir los descriptores finales de vlad
      image_struct* image_aux=(image_struct*)malloc(sizeof(image_struct));

      start_crono( "DSIFT" ) ;

      if(dim_sift_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_sift_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_dsift = dsift_features ( image_aux, get_segmentation_data(seg), (int*)get_command_arguments_dsift_parameters(command_arguments) ) ;
      } else{
        model_dsift = dsift_features ( image, get_segmentation_data(seg), (int*)get_command_arguments_dsift_parameters(command_arguments) ) ;
      }


      //R4
      raw_features = (unsigned int*)malloc(model_dsift.total_descriptors * dim_sift_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_dsift.num_segments;i++){
        for(unsigned int j=0;j<model_dsift.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_sift_descriptor;k++){
            raw_features[partial_sum*dim_sift_descriptor + k] = (int) model_dsift.descriptors[i][j].desc[k];
          }
          partial_sum ++;
        }
      }


      stop_crono ( ) ;



      start_crono( "KMEANS" ) ;

      //R4
      kmeans( raw_features , model_dsift.total_descriptors, dim_sift_descriptor, params,  &model_kmeans ) ;

      stop_crono ( ) ;


      start_crono( "VLAD" ) ;

      //R4 PCA
      double * data_aux = vlad_sift( model_dsift, dim_sift_descriptor, get_image_width(image), get_image_height(image), model_kmeans, H1 , K) ;


      Eigen::MatrixXf pca_data_matrix(get_segmentation_number_segments(seg), model_kmeans.K * dim_sift_descriptor );
      for(unsigned int i=0; i<get_segmentation_number_segments(seg); i++){
      	for(unsigned int j=0; j<dim_sift_descriptor*model_kmeans.K; j++){
      		pca_data_matrix(i,j) = data_aux[i*dim_sift_descriptor+j];
      	}
      }

      pca_t<float> pca;
      pca.set_input(pca_data_matrix);
      pca.compute();

      double* part_vlad = (double *)calloc(get_segmentation_number_segments(seg)*R4,sizeof(double));
      for(unsigned int k=0;k<get_segmentation_number_segments(seg);k++){
      	for( int i=0;i<R4;i++){
          part_vlad[k*R4+i]=0;
          for(unsigned int j=0;j<dim_sift_descriptor;j++){
      		    part_vlad[k*R4+i] = part_vlad[k*R4+i] + pca_data_matrix(k,j)*pca.get_eigen_vectors()(j,i);
          }
      	}
      }

      // //R4 Media
      // double * data_aux = vlad_sift( model_dsift, dim_sift_descriptor, get_image_width(image), get_image_height(image), model_kmeans, H1 , K) ;;
      //
      // int n=R4, x=K*dim_sift_descriptor;
      // int* parts =force_integer_splits(n, x);
      // int iacum;
      // double* part_vlad = (double *)calloc(H1*n,sizeof(double));
      // for(int i=0;i<H1;i++){
      //   iacum=0;
      //   for(int j=0;j<n;j++){
      //     for(int pi=0;pi<parts[j];pi++){
      //       part_vlad[i*n+j] = part_vlad[i*n+j] + data_aux[i*dim_sift_descriptor*model_kmeans.K+iacum];
      //       iacum++;
      //     }
      //     part_vlad[i*n+j] = part_vlad[i*n+j] / parts[j];
      //   }
      // }

      //Concatenando pixel al descriptor vlad
      //double* central_per_segment = get_central_pixel_per_segment(seg, image, get_segmentation_number_segments(seg));
      int* central_per_segment = get_means_per_segment(seg, image, get_segmentation_number_segments(seg));
      data = (double*) malloc(get_segmentation_number_segments(seg) * (R4+get_image_bands(image)) * sizeof(double));
      for(unsigned int i=0; i < get_segmentation_number_segments(seg); i++){
        for(unsigned int j=0; j < (R4+get_image_bands(image)); j++){
          if(j < (unsigned int)(R4)){
            data[i*(R4+get_image_bands(image))+j] = part_vlad[i*(R4)+j];
          }else{
            data[i*(R4+get_image_bands(image))+j] = central_per_segment[i*get_image_bands(image) + (j-(R4)) ];
          }
        }
      }

      dim = R4 + get_image_bands(image);



      destroy_kmeans_model( model_kmeans ) ;

      stop_crono ( ) ;

      break;}





    case 11:{ //METODO 11: DSIFT, GMM y fisher vectors (descrs)
      gmm_parameter_t gmm_params;
      descriptor_model_t model_dsift ;
      gmm_model_t model_gmm;
      int partial_sum = 0;
      unsigned int * raw_features;
      image_struct* image_aux = (image_struct*)malloc(sizeof(image_struct));
      int R4=5, d=0;


      start_crono( "DSIFT" ) ;

      if(dim_sift_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_sift_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_dsift = dsift_features ( image_aux, get_segmentation_data(seg), (int*)get_command_arguments_dsift_parameters(command_arguments) ) ;
      } else{
        model_dsift = dsift_features ( image, get_segmentation_data(seg), (int*)get_command_arguments_dsift_parameters(command_arguments) ) ;
      }

      // preparacion de datos de sift para kmeans
      raw_features = (unsigned int*)malloc(model_dsift.total_descriptors * dim_sift_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_dsift.num_segments;i++){
        for(unsigned int j=0;j<model_dsift.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_sift_descriptor;k++){
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



      start_crono( "FisherVectors" ) ;

      double* data_aux=NULL;
      if(dim_sift_descriptor > get_image_bands(image)){
        reduce_dim_after_clustering(image, NULL, &model_gmm, dim_sift_descriptor, get_command_arguments_reduction_method(command_arguments)*2);
        data_aux = fishervectors_features( get_image_data(image), seg, model_gmm )  ;
	      d=model_gmm.dimensions;
      } else{
        data_aux = fishervectors_features( get_image_data(image_aux), seg, model_gmm )  ;
	      d=dim_sift_descriptor;
      }


      //R4 PCA
      Eigen::MatrixXf pca_data_matrix(get_segmentation_number_segments(seg), model_gmm.centers * d );
      for(unsigned int i=0; i<get_segmentation_number_segments(seg); i++){
      	for( int j=0; j<d*model_gmm.centers; j++){
      		pca_data_matrix(i,j) = data_aux[i*d+j];
      	}
      }

      pca_t<float> pca;
      pca.set_input(pca_data_matrix);
      pca.compute();

      double* part_fisher = (double *)calloc(get_segmentation_number_segments(seg)*R4,sizeof(double));
      for(unsigned int k=0;k<get_segmentation_number_segments(seg);k++){
      	for( int i=0;i<R4;i++){
          part_fisher[k*R4+i]=0;
          for(int j=0;j<d;j++){
      		    part_fisher[k*R4+i] = part_fisher[k*R4+i] + pca_data_matrix(k,j)*pca.get_eigen_vectors()(j,i);
          }
      	}
      }

      // //R4 Media todo:cambiar sift por gmmdimensions
      //
      // int n=R4, x=K*dim_sift_descriptor;
      // int* parts =force_integer_splits(n, x);
      // int iacum;
      // double* part_fisher = (double *)calloc(H1*n,sizeof(double));
      // for(int i=0;i<H1;i++){
      //   iacum=0;
      //   for(int j=0;j<n;j++){
      //     for(int pi=0;pi<parts[j];pi++){
      //       part_fisher[i*n+j] = part_fisher[i*n+j] + data_aux[i*dim_sift_descriptor*model_kmeans.K+iacum];
      //       iacum++;
      //     }
      //     part_fisher[i*n+j] = part_fisher[i*n+j] / parts[j];
      //   }
      // }

      //Concatenando pixel al descriptor vlad
      //double* central_per_segment = get_central_pixel_per_segment(seg, image, get_segmentation_number_segments(seg));
      int* central_per_segment = get_means_per_segment(seg, image, get_segmentation_number_segments(seg));
      data = (double*) malloc(get_segmentation_number_segments(seg) * (R4+get_image_bands(image)) * sizeof(double));
      for(unsigned int i=0; i < get_segmentation_number_segments(seg); i++){
        for(unsigned int j=0; j < (R4+get_image_bands(image)); j++){
          if(j < (unsigned int)(R4)){
            data[i*(R4+get_image_bands(image))+j] = part_fisher[i*(R4)+j];
          }else{
            data[i*(R4+get_image_bands(image))+j] = central_per_segment[i*get_image_bands(image) + (j-(R4)) ];
          }
        }
      }

      stop_crono ( ) ;

      dim = R4 + get_image_bands(image);

      break;}





    case 12:{ // METODO 12: LIOP, Kmeans y VLAD
      kmeans_parameter_t params ;
      kmeans_model_t model_kmeans ;
      descriptor_model_t model_liop ;
      int H1, V1 ;
      int partial_sum = 0;
      unsigned int* raw_features;
      image_struct* image_aux = (image_struct*)malloc(sizeof(image_struct));
      unsigned int dim_liop_descriptor = get_command_arguments_liop_parameters(command_arguments)[2] * factorial(get_command_arguments_liop_parameters(command_arguments)[1]);


      start_crono( "LIOP" ) ;

      if(dim_liop_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_liop_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_liop = liop_features ( image_aux, get_segmentation_data(seg), NULL, (float*)get_command_arguments_liop_parameters(command_arguments) ) ;
      } else{
        model_liop = liop_features ( image, get_segmentation_data(seg), NULL, (float*)get_command_arguments_liop_parameters(command_arguments) ) ;
      }

      // preparacion de datos de liop para kmeans
      raw_features = (unsigned int*)malloc(model_liop.total_descriptors * dim_liop_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_liop.num_segments;i++){
        for(unsigned int j=0;j<model_liop.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_liop_descriptor;k++){
            raw_features[partial_sum*dim_liop_descriptor + k] = (int) 100000 * model_liop.descriptors[i][j].desc[k];
          }
          partial_sum ++;
        }
      }

      stop_crono ( ) ;




      start_crono( "KMEANS" ) ;

      kmeans( raw_features , model_liop.total_descriptors, dim_liop_descriptor, params,  &model_kmeans ) ;

      stop_crono ( ) ;




      start_crono( "VLAD" ) ;

      if(dim_liop_descriptor > get_image_bands(image)){
        reduce_dim_after_clustering(image, &model_kmeans, NULL, dim_liop_descriptor, get_command_arguments_reduction_method(command_arguments));
        data = vlad( get_image_data(image), get_segmentation_data(seg), get_image_bands(image), get_image_width(image), get_image_height(image), model_kmeans, H1 , V1, K) ;
        dim = K*get_image_bands(image);
      } else{
        data = vlad( get_image_data(image_aux), get_segmentation_data(seg), dim_liop_descriptor, get_image_width(image), get_image_height(image), model_kmeans, H1 , V1, K) ;
        dim = K*get_image_bands(image_aux);
      }

      destroy_kmeans_model( model_kmeans ) ;

      stop_crono ( ) ;


      break;}





    case 13:{ //METODO 13: LIOP, GMM y fisher vectors
      gmm_parameter_t gmm_params;
      descriptor_model_t model_liop ;
      gmm_model_t model_gmm;
      int partial_sum = 0;
      unsigned int * raw_features;
      image_struct* image_aux = (image_struct*)malloc(sizeof(image_struct));
      unsigned int dim_liop_descriptor = get_command_arguments_liop_parameters(command_arguments)[2] * factorial(get_command_arguments_liop_parameters(command_arguments)[1]);


      start_crono( "LIOP" ) ;

      if(dim_liop_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_liop_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_liop = liop_features ( image_aux, get_segmentation_data(seg), NULL, (float*)get_command_arguments_liop_parameters(command_arguments) ) ;
      } else{
        model_liop = liop_features ( image, get_segmentation_data(seg), NULL, (float*)get_command_arguments_liop_parameters(command_arguments) ) ;
      }

      // preparacion de datos liop para kmeans
      raw_features = (unsigned int*)malloc(model_liop.total_descriptors * dim_liop_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_liop.num_segments;i++){
        for(unsigned int j=0;j<model_liop.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_liop_descriptor;k++){
            raw_features[partial_sum*dim_liop_descriptor + k] = (int) model_liop.descriptors[i][j].desc[k];
          }
          partial_sum ++;
        }
      }

      stop_crono ( ) ;




      start_crono( "GMM" ) ;
      gmm_params.dimensions = dim_liop_descriptor;
      gmm_params.numPixels = model_liop.total_descriptors;
      model_gmm = gmm ( raw_features, gmm_params ) ;

      stop_crono ( ) ;




      start_crono( "FisherVectors" ) ;

      if(dim_liop_descriptor > get_image_bands(image)){
        reduce_dim_after_clustering(image, NULL, &model_gmm, dim_liop_descriptor, get_command_arguments_reduction_method(command_arguments)*2);
        data = fishervectors_features( get_image_data(image), seg, model_gmm )  ;
        dim = K*get_image_bands(image);
      } else{
        data = fishervectors_features( get_image_data(image_aux), seg, model_gmm )  ;
        dim = K*get_image_bands(image_aux);
      }

      stop_crono ( ) ;

      break;}





    case 14:{ // METODO 14: LIOP, Kmeans y VLAD (descs)
      kmeans_parameter_t params ;
      kmeans_model_t model_kmeans ;
      descriptor_model_t model_liop ;
      int H1;
      int partial_sum = 0;
      int R4=5;
      unsigned int* raw_features;
      image_struct* image_aux = (image_struct*)malloc(sizeof(image_struct));
      unsigned int dim_liop_descriptor = get_command_arguments_liop_parameters(command_arguments)[2] * factorial(get_command_arguments_liop_parameters(command_arguments)[1]);


      start_crono( "LIOP" ) ;

      if(dim_liop_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_liop_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_liop = liop_features ( image_aux, get_segmentation_data(seg), NULL, (float*)get_command_arguments_liop_parameters(command_arguments) ) ;
      } else{
        model_liop = liop_features ( image, get_segmentation_data(seg), NULL, (float*)get_command_arguments_liop_parameters(command_arguments) ) ;
      }

      // preparacion de datos de liop para kmeans
      raw_features = (unsigned int*)malloc(model_liop.total_descriptors * dim_liop_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_liop.num_segments;i++){
        for(unsigned int j=0;j<model_liop.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_liop_descriptor;k++){
            raw_features[partial_sum*dim_liop_descriptor + k] = (int) 100000 * model_liop.descriptors[i][j].desc[k];
          }
          partial_sum ++;
        }
      }

      stop_crono ( ) ;




      start_crono( "KMEANS" ) ;

      kmeans( raw_features , model_liop.total_descriptors, dim_liop_descriptor, params,  &model_kmeans ) ;

      stop_crono ( ) ;




      start_crono( "VLAD" ) ;

      double * data_aux = vlad_sift( model_liop, dim_liop_descriptor, get_image_width(image), get_image_height(image), model_kmeans, H1 , K) ;

      Eigen::MatrixXf pca_data_matrix(get_segmentation_number_segments(seg), model_kmeans.K * dim_liop_descriptor );
      for(unsigned int i=0; i<get_segmentation_number_segments(seg); i++){
        for(unsigned int j=0; j<dim_liop_descriptor*model_kmeans.K; j++){
          pca_data_matrix(i,j) = data_aux[i*dim_liop_descriptor+j];
        }
      }

      pca_t<float> pca;
      pca.set_input(pca_data_matrix);
      pca.compute();

      double* part_vlad = (double *)calloc(get_segmentation_number_segments(seg)*R4,sizeof(double));
      for(unsigned int k=0;k<get_segmentation_number_segments(seg);k++){
        for( int i=0;i<R4;i++){
          part_vlad[k*R4+i]=0;
          for(unsigned int j=0;j<dim_liop_descriptor;j++){
              part_vlad[k*R4+i] = part_vlad[k*R4+i] + pca_data_matrix(k,j)*pca.get_eigen_vectors()(j,i);
          }
        }
      }

      //Concatenando pixel al descriptor vlad
      int* central_per_segment = get_means_per_segment(seg, image, get_segmentation_number_segments(seg));
      data = (double*) malloc(get_segmentation_number_segments(seg) * (R4+get_image_bands(image)) * sizeof(double));
      for(unsigned int i=0; i < get_segmentation_number_segments(seg); i++){
        for(unsigned int j=0; j < (R4+get_image_bands(image)); j++){
          if(j < (unsigned int)(R4)){
            data[i*(R4+get_image_bands(image))+j] = part_vlad[i*(R4)+j];
          }else{
            data[i*(R4+get_image_bands(image))+j] = central_per_segment[i*get_image_bands(image) + (j-(R4)) ];
          }
        }
      }

      dim = R4 + get_image_bands(image);

      destroy_kmeans_model( model_kmeans ) ;

      stop_crono ( ) ;


      break;}





    case 15:{ //METODO 15: LIOP, GMM y fisher vectors (descs)
      gmm_parameter_t gmm_params;
      descriptor_model_t model_liop ;
      gmm_model_t model_gmm;
      int R4=5, d=0;
      int partial_sum = 0;
      unsigned int * raw_features;
      image_struct* image_aux = (image_struct*)malloc(sizeof(image_struct));
      unsigned int dim_liop_descriptor = get_command_arguments_liop_parameters(command_arguments)[2] * factorial(get_command_arguments_liop_parameters(command_arguments)[1]);


      start_crono( "LIOP" ) ;

      if(dim_liop_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_liop_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_liop = liop_features ( image_aux, get_segmentation_data(seg), NULL, (float*)get_command_arguments_liop_parameters(command_arguments) ) ;
      } else{
        model_liop = liop_features ( image, get_segmentation_data(seg), NULL, (float*)get_command_arguments_liop_parameters(command_arguments) ) ;
      }

      // preparacion de datos liop para kmeans
      raw_features = (unsigned int*)malloc(model_liop.total_descriptors * dim_liop_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_liop.num_segments;i++){
        for(unsigned int j=0;j<model_liop.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_liop_descriptor;k++){
            raw_features[partial_sum*dim_liop_descriptor + k] = (int) model_liop.descriptors[i][j].desc[k];
          }
          partial_sum ++;
        }
      }

      stop_crono ( ) ;




      start_crono( "GMM" ) ;
      gmm_params.dimensions = dim_liop_descriptor;
      gmm_params.numPixels = model_liop.total_descriptors;
      model_gmm = gmm ( raw_features, gmm_params ) ;

      stop_crono ( ) ;




      start_crono( "FisherVectors" ) ;

      double* data_aux=NULL;
      if(dim_liop_descriptor > get_image_bands(image)){
        reduce_dim_after_clustering(image, NULL, &model_gmm, dim_liop_descriptor, get_command_arguments_reduction_method(command_arguments)*2);
        data_aux = fishervectors_features( get_image_data(image), seg, model_gmm )  ;
        d=model_gmm.dimensions;
      } else{
        data_aux = fishervectors_features( get_image_data(image_aux), seg, model_gmm )  ;
        d=dim_liop_descriptor;
      }

      //R4 PCA
      Eigen::MatrixXf pca_data_matrix(get_segmentation_number_segments(seg), model_gmm.centers * d );
      for(unsigned int i=0; i<get_segmentation_number_segments(seg); i++){
        for( int j=0; j<d*model_gmm.centers; j++){
          pca_data_matrix(i,j) = data_aux[i*d+j];
        }
      }

      pca_t<float> pca;
      pca.set_input(pca_data_matrix);
      pca.compute();

      double* part_fisher = (double *)calloc(get_segmentation_number_segments(seg)*R4,sizeof(double));
      for(unsigned int k=0;k<get_segmentation_number_segments(seg);k++){
        for( int i=0;i<R4;i++){
          part_fisher[k*R4+i]=0;
          for(int j=0;j<d;j++){
              part_fisher[k*R4+i] = part_fisher[k*R4+i] + pca_data_matrix(k,j)*pca.get_eigen_vectors()(j,i);
          }
        }
      }

      //Concatenando pixel al descriptor vlad
      int* central_per_segment = get_means_per_segment(seg, image, get_segmentation_number_segments(seg));
      data = (double*) malloc(get_segmentation_number_segments(seg) * (R4+get_image_bands(image)) * sizeof(double));
      for(unsigned int i=0; i < get_segmentation_number_segments(seg); i++){
        for(unsigned int j=0; j < (R4+get_image_bands(image)); j++){
          if(j < (unsigned int)(R4)){
            data[i*(R4+get_image_bands(image))+j] = part_fisher[i*(R4)+j];
          }else{
            data[i*(R4+get_image_bands(image))+j] = central_per_segment[i*get_image_bands(image) + (j-(R4)) ];
          }
        }
      }

      dim = R4 + get_image_bands(image);

      stop_crono ( ) ;

      break;}





    case 16:{ // METODO 16: HOG, Kmeans y VLAD
      kmeans_parameter_t params ;
      kmeans_model_t model_kmeans ;
      descriptor_model_t model_hog ;
      int H1, V1 ;
      int partial_sum = 0;
      unsigned int* raw_features;
      image_struct* image_aux = (image_struct*)malloc(sizeof(image_struct));
      unsigned int dim_hog_descriptor = get_command_arguments_hog_parameters(command_arguments)[0]*4;


      start_crono( "HOG" ) ;

      if(dim_hog_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_hog_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_hog = hog_features ( image_aux, get_segmentation_data(seg), (int*)get_command_arguments_hog_parameters(command_arguments) ) ;
      } else{
        model_hog = hog_features ( image, get_segmentation_data(seg), (int*)get_command_arguments_hog_parameters(command_arguments) );
      }


      // preparacion de datos de liop para kmeans
      raw_features = (unsigned int*)malloc(model_hog.total_descriptors * dim_hog_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_hog.num_segments;i++){
        for(unsigned int j=0;j<model_hog.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_hog_descriptor;k++){
            raw_features[partial_sum*dim_hog_descriptor + k] = (int) 100000 * model_hog.descriptors[i][j].desc[k];
          }
          partial_sum ++;
        }
      }

      stop_crono ( ) ;




      start_crono( "KMEANS" ) ;

      kmeans( raw_features , model_hog.total_descriptors, dim_hog_descriptor, params,  &model_kmeans ) ;

      stop_crono ( ) ;




      start_crono( "VLAD" ) ;

      if(dim_hog_descriptor > get_image_bands(image)){
        reduce_dim_after_clustering(image, &model_kmeans, NULL, dim_hog_descriptor, get_command_arguments_reduction_method(command_arguments));
        data = vlad( get_image_data(image), get_segmentation_data(seg), get_image_bands(image), get_image_width(image), get_image_height(image), model_kmeans, H1 , V1, K) ;
        dim = K*get_image_bands(image);
      } else{
        data = vlad( get_image_data(image_aux), get_segmentation_data(seg), dim_hog_descriptor, get_image_width(image), get_image_height(image), model_kmeans, H1 , V1, K) ;
        dim = K*get_image_bands(image_aux);
      }

      destroy_kmeans_model( model_kmeans ) ;

      stop_crono ( ) ;


      break;}





    case 17:{ //METODO 17: HOG, GMM y fisher vectors
      gmm_parameter_t gmm_params;
      descriptor_model_t model_hog ;
      gmm_model_t model_gmm;
      int partial_sum = 0;
      unsigned int * raw_features;
      image_struct* image_aux = (image_struct*)malloc(sizeof(image_struct));
      unsigned int dim_hog_descriptor = get_command_arguments_hog_parameters(command_arguments)[0]*4;


      start_crono( "HOG" ) ;

      if(dim_hog_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_hog_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_hog = hog_features ( image_aux, get_segmentation_data(seg), (int*)get_command_arguments_hog_parameters(command_arguments) ) ;
      } else{
        model_hog = hog_features ( image, get_segmentation_data(seg), (int*)get_command_arguments_hog_parameters(command_arguments) );
      }


      // preparacion de datos hog para kmeans
      raw_features = (unsigned int*)malloc(model_hog.total_descriptors * dim_hog_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_hog.num_segments;i++){
        for(unsigned int j=0;j<model_hog.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_hog_descriptor;k++){
            raw_features[partial_sum*dim_hog_descriptor + k] = (int) model_hog.descriptors[i][j].desc[k];
          }
          partial_sum ++;
        }
      }

      stop_crono ( ) ;




      start_crono( "GMM" ) ;

      gmm_params.dimensions = dim_hog_descriptor;
      gmm_params.numPixels = model_hog.total_descriptors;
      model_gmm = gmm ( raw_features, gmm_params ) ;

      stop_crono ( ) ;




      start_crono( "FisherVectors" ) ;

      if(dim_hog_descriptor > get_image_bands(image)){
        reduce_dim_after_clustering(image, NULL, &model_gmm, dim_hog_descriptor, get_command_arguments_reduction_method(command_arguments)*2);
        data = fishervectors_features( get_image_data(image), seg, model_gmm )  ;
        dim = K*get_image_bands(image);
      } else{
        data = fishervectors_features( get_image_data(image_aux), seg, model_gmm )  ;
        dim = K*get_image_bands(image_aux);
      }

      stop_crono ( ) ;

      break;}





    case 18:{ // METODO 18: HOG, Kmeans y VLAD (descs)
      kmeans_parameter_t params ;
      kmeans_model_t model_kmeans ;
      descriptor_model_t model_hog ;
      int H1 ;
      int partial_sum = 0;
      unsigned int* raw_features;
      image_struct* image_aux = (image_struct*)malloc(sizeof(image_struct));
      unsigned int dim_hog_descriptor = get_command_arguments_hog_parameters(command_arguments)[0]*4;
      int R4=5;


      start_crono( "HOG" ) ;

      if(dim_hog_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_hog_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_hog = hog_features ( image_aux, get_segmentation_data(seg), (int*)get_command_arguments_hog_parameters(command_arguments) ) ;
      } else{
        model_hog = hog_features ( image, get_segmentation_data(seg), (int*)get_command_arguments_hog_parameters(command_arguments) );
      }


      // preparacion de datos de liop para kmeans
      raw_features = (unsigned int*)malloc(model_hog.total_descriptors * dim_hog_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_hog.num_segments;i++){
        for(unsigned int j=0;j<model_hog.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_hog_descriptor;k++){
            raw_features[partial_sum*dim_hog_descriptor + k] = (int) 100000 * model_hog.descriptors[i][j].desc[k];
          }
          partial_sum ++;
        }
      }

      stop_crono ( ) ;




      start_crono( "KMEANS" ) ;

      kmeans( raw_features , model_hog.total_descriptors, dim_hog_descriptor, params,  &model_kmeans ) ;

      stop_crono ( ) ;




      start_crono( "VLAD" ) ;

      double * data_aux = vlad_sift( model_hog, dim_hog_descriptor, get_image_width(image), get_image_height(image), model_kmeans, H1 , K) ;

      Eigen::MatrixXf pca_data_matrix(get_segmentation_number_segments(seg), model_kmeans.K * dim_hog_descriptor );
      for(unsigned int i=0; i<get_segmentation_number_segments(seg); i++){
        for(unsigned int j=0; j<dim_hog_descriptor*model_kmeans.K; j++){
          pca_data_matrix(i,j) = data_aux[i*dim_hog_descriptor+j];
        }
      }

      pca_t<float> pca;
      pca.set_input(pca_data_matrix);
      pca.compute();

      double* part_vlad = (double *)calloc(get_segmentation_number_segments(seg)*R4,sizeof(double));
      for(unsigned int k=0;k<get_segmentation_number_segments(seg);k++){
        for( int i=0;i<R4;i++){
          part_vlad[k*R4+i]=0;
          for(unsigned int j=0;j<dim_hog_descriptor;j++){
              part_vlad[k*R4+i] = part_vlad[k*R4+i] + pca_data_matrix(k,j)*pca.get_eigen_vectors()(j,i);
          }
        }
      }

      //Concatenando pixel al descriptor vlad
      int* central_per_segment = get_means_per_segment(seg, image, get_segmentation_number_segments(seg));
      data = (double*) malloc(get_segmentation_number_segments(seg) * (R4+get_image_bands(image)) * sizeof(double));
      for(unsigned int i=0; i < get_segmentation_number_segments(seg); i++){
        for(unsigned int j=0; j < (R4+get_image_bands(image)); j++){
          if(j < (unsigned int)(R4)){
            data[i*(R4+get_image_bands(image))+j] = part_vlad[i*(R4)+j];
          }else{
            data[i*(R4+get_image_bands(image))+j] = central_per_segment[i*get_image_bands(image) + (j-(R4)) ];
          }
        }
      }

      dim = R4 + get_image_bands(image);

      destroy_kmeans_model( model_kmeans ) ;

      stop_crono ( ) ;


      break;}





    case 19:{ //METODO 19: HOG, GMM y fisher vectors (descs)
      int R4=5, d=0;
      gmm_parameter_t gmm_params;
      gmm_model_t model_gmm;
      descriptor_model_t model_hog ;
      int partial_sum = 0;
      unsigned int* raw_features;
      image_struct* image_aux = (image_struct*)malloc(sizeof(image_struct));
      unsigned int dim_hog_descriptor = get_command_arguments_hog_parameters(command_arguments)[0]*4;


      start_crono( "HOG" ) ;

      if(dim_hog_descriptor < get_image_bands(image)){
        reduce_dim_before_descriptors(image, image_aux, dim_hog_descriptor, get_command_arguments_reduction_method(command_arguments), error);
        model_hog = hog_features ( image_aux, get_segmentation_data(seg), (int*)get_command_arguments_hog_parameters(command_arguments) ) ;
      } else{
        model_hog = hog_features ( image, get_segmentation_data(seg), (int*)get_command_arguments_hog_parameters(command_arguments) );
      }


      // preparacion de datos liop para kmeans
      raw_features = (unsigned int*)malloc(model_hog.total_descriptors * dim_hog_descriptor * sizeof(unsigned int));
      for(int i=0;i<model_hog.num_segments;i++){
        for(unsigned int j=0;j<model_hog.descriptors[i].size();j++){
          for(unsigned int k=0;k<dim_hog_descriptor;k++){
            raw_features[partial_sum*dim_hog_descriptor + k] = (int) model_hog.descriptors[i][j].desc[k];
          }
          partial_sum ++;
        }
      }

      stop_crono ( ) ;




      start_crono( "GMM" ) ;
      gmm_params.dimensions = dim_hog_descriptor;
      gmm_params.numPixels = model_hog.total_descriptors;
      model_gmm = gmm ( raw_features, gmm_params ) ;

      stop_crono ( ) ;




      start_crono( "FisherVectors" ) ;

      double* data_aux=NULL;
      if(dim_hog_descriptor > get_image_bands(image)){
        reduce_dim_after_clustering(image, NULL, &model_gmm, dim_hog_descriptor, get_command_arguments_reduction_method(command_arguments)*2);
        data_aux = fishervectors_features( get_image_data(image), seg, model_gmm )  ;
        d=model_gmm.dimensions;
      } else{
        data_aux = fishervectors_features( get_image_data(image_aux), seg, model_gmm )  ;
        d=dim_hog_descriptor;
      }

      //R4 PCA
      Eigen::MatrixXf pca_data_matrix(get_segmentation_number_segments(seg), model_gmm.centers * d );
      for(unsigned int i=0; i<get_segmentation_number_segments(seg); i++){
        for( int j=0; j<d*model_gmm.centers; j++){
          pca_data_matrix(i,j) = data_aux[i*d+j];
        }
      }

      pca_t<float> pca;
      pca.set_input(pca_data_matrix);
      pca.compute();

      double* part_fisher = (double *)calloc(get_segmentation_number_segments(seg)*R4,sizeof(double));
      for(unsigned int k=0;k<get_segmentation_number_segments(seg);k++){
        for( int i=0;i<R4;i++){
          part_fisher[k*R4+i]=0;
          for(int j=0;j<d;j++){
              part_fisher[k*R4+i] = part_fisher[k*R4+i] + pca_data_matrix(k,j)*pca.get_eigen_vectors()(j,i);
          }
        }
      }

      //Concatenando pixel al descriptor vlad
      int* central_per_segment = get_means_per_segment(seg, image, get_segmentation_number_segments(seg));
      data = (double*) malloc(get_segmentation_number_segments(seg) * (R4+get_image_bands(image)) * sizeof(double));
      for(unsigned int i=0; i < get_segmentation_number_segments(seg); i++){
        for(unsigned int j=0; j < (R4+get_image_bands(image)); j++){
          if(j < (unsigned int)(R4)){
            data[i*(R4+get_image_bands(image))+j] = part_fisher[i*(R4)+j];
          }else{
            data[i*(R4+get_image_bands(image))+j] = central_per_segment[i*get_image_bands(image) + (j-(R4)) ];
          }
        }
      }

      dim = R4 + get_image_bands(image);

      stop_crono ( ) ;

      break;}



    default:{
      print_error((char*)"Texture method not recognized");
      exit(EXIT_FAILURE);}
      break;}





  set_descriptors_dim_descriptors(descriptors, dim, error);
  set_descriptors_number_descriptors(descriptors, get_segmentation_number_segments(seg), error);

  data_std = standardize(data, dim, get_segmentation_number_segments(seg), error);
  set_descriptors_data(descriptors, data_std, error);

  free(data);
  free(data_std);

  //labels = get_labels_per_segment_majority_voting(seg, gt_train, get_descriptors_number_descriptors(descriptors));
  labels = get_labels_per_segment_central_pixels(seg, gt_train, get_descriptors_number_descriptors(descriptors));
  set_descriptors_labels(descriptors, labels, error);
  free(labels);

  return descriptors;
}
