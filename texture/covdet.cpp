

#include "covdet.h"




detector_model_t *covdet_keypoints ( image_struct * image, float* parameters )
{

  /*********** variables ******************/
  vl_size w = get_image_width(image), h = get_image_height(image), max_value =0, min_value=10000000;
  float *fdata = 0 ;
  detector_model_t *output;
  // -8 4 15 7 3 1 (para SIFT RAW)
  VlCovDetMethod method = (VlCovDetMethod)parameters[0];
    //VL_COVDET_METHOD_NUM = 0
    //VL_COVDET_METHOD_DOG = 1
    //VL_COVDET_METHOD_HESSIAN -
    //VL_COVDET_METHOD_HESSIAN_LAPLACE --
    //VL_COVDET_METHOD_HARRIS_LAPLACE ---
    //VL_COVDET_METHOD_MULTISCALE_HESSIAN
    //VL_COVDET_METHOD_MULTISCALE_HARRIS

  vl_size patchResolution = (vl_size)parameters[1];
    //= 15 ;
  vl_size ps = 2*patchResolution + 1 ;
  double patchRelativeExtent = (double)parameters[2];
    //7.5 ;
  double patchRelativeSmoothing = (double)parameters[3];
    // = 1 ;
  //int NOCT = 8;
  double boundaryMargin = (double)parameters[4];
    //= 2;
  VlCovDet * covdet = vl_covdet_new( method ) ;

  VlRand rand ;
  vl_rand_init (&rand) ;
	vl_rand_seed (&rand,  1000) ;


  output = (detector_model_t*) malloc(get_image_bands(image)*sizeof(detector_model_t));

  for(unsigned int i=0;i<get_image_bands(image);i++){ //para cada dimensión
    printf("\n\n\t** Band %d **\n", i);

    /*********** rescaling: [0,255] ******************/
    find_maxmin(get_image_data(image)+(i*get_image_width(image)*get_image_height(image)),   get_image_width(image)*get_image_height(image),  &min_value,  &max_value);

    fdata = (float *) malloc(get_image_width(image)*get_image_height(image) * sizeof ((double) get_image_data(image)[0]) * sizeof (float)) ;

    //rescale: rescales the values in the range [0.0, 255.0] as the algorithm is tuned for PGM images
    for(unsigned int j=0; j<get_image_width(image)*get_image_height(image);j++){
      fdata[j] = (float) 0 + ( (float)((get_image_data(image)[((i*get_image_width(image)*get_image_height(image)) + j)] - min_value)*(255-0)) / (float)(max_value-min_value) );
      //printf(" %f  ", fdata[j]);
      //fdata[j] = (float) get_image_data(image)[((i*get_image_width(image)*get_image_height(image)) + j)];
    }


    /************* get keypoint patches ************/

    int size = 1000;
    if(get_image_width(image) > (unsigned int) size && get_image_height(image) > (unsigned int) size){

      float* patchImage;
      //float* patches; //std::vector<double> patches ;
      //int* coords; //std::vector<int> coords ;
      output[i].num_patches = 0;

      for(unsigned int a=0; a<w/size; a++){
        for(unsigned int b=0; b<h/size; b++){
          patchImage = (float*)vl_malloc(sizeof(float)*size*size);

          for(int p1 = 0; p1 < (signed)size; p1++){
            for(int p2 = 0; p2 < (signed)size; p2++){
              patchImage[ p1*size+p2 ] = fdata[ (a*size+p1)*w + (b*size)+p2 ];
            }
          }

          vl_covdet_put_image(covdet, patchImage, size, size) ;
          vl_covdet_detect(covdet) ;
          vl_covdet_drop_features_outside (covdet, boundaryMargin) ;
          vl_covdet_extract_affine_shape(covdet) ;
          vl_covdet_extract_orientations(covdet) ;
          vl_size numFeatures = vl_covdet_get_num_features(covdet) ;
          VlCovDetFeature const * feature = (const VlCovDetFeature*) vl_covdet_get_features(covdet) ;

          output[i].num_patches += numFeatures;
          output[i].dim_patches = ps*ps;
          printf("id %lld num %d  dim %d \n", (a*(w/size)+b), output[i].num_patches, output[i].dim_patches);

          for (unsigned int f = 0 ; f < numFeatures ; ++f) {
            float * patch = (float*)malloc(ps*ps*sizeof(float)) ;

            vl_covdet_extract_patch_for_frame(covdet, patch, patchResolution, patchRelativeExtent, patchRelativeSmoothing, feature[f].frame) ;

            free(patch);
          }


        }
      }

    } else {
      // create a detector object


      // set various parameters (optional)
      //vl_covdet_set_first_octave(covdet, -1) ; // start by doubling the image resolution
      //vl_covdet_set_octave_resolution(covdet, octaveResolution) ;
      //vl_covdet_set_peak_threshold(covdet, peakThreshold) ;
      //vl_covdet_set_edge_threshold(covdet, edgeThreshold) ;
      //vl_covdet_set_num_octaves(covdet, NOCT);

      // process the image and run the detector
      vl_covdet_put_image(covdet, fdata, w, h) ;
      vl_covdet_detect(covdet) ;

      // drop features on the margin (optional)
      vl_covdet_drop_features_outside (covdet, boundaryMargin) ;
      // compute the affine shape of the features (optional)
      vl_covdet_extract_affine_shape(covdet);
      // compute the orientation of the features (optional)
      vl_covdet_extract_orientations(covdet) ;

      // get feature frames back
      vl_size numFeatures = vl_covdet_get_num_features(covdet) ;
      VlCovDetFeature const * feature = (const VlCovDetFeature*) vl_covdet_get_features(covdet) ;

      output[i].num_patches = numFeatures;
      output[i].dim_patches = ps*ps;
      output[i].coords = (int*) malloc(numFeatures*2*sizeof(int));
      output[i].patches = (double*) malloc(numFeatures*ps*ps*sizeof(double));


      // get normalized feature appearance patches (optional)
      for (unsigned int f = 0 ; f < numFeatures ; ++f) {
        float * patch = (float*)malloc(ps*ps*sizeof(float)) ;
        vl_covdet_extract_patch_for_frame(covdet,
                              patch,
                              patchResolution,
                              patchRelativeExtent,
                              patchRelativeSmoothing,
                              feature[f].frame) ;

        output[i].coords[2*f] = (int)round(feature[f].frame.x);
        output[i].coords[2*f+1] = (int)round(feature[f].frame.y);
        //printf("%f %f \n", feature[i].frame.x, feature[i].frame.y);

        for (unsigned int j = 0 ; j < ps*ps ; ++j) {
          output[i].patches[f*(ps*ps)+j] = patch[j];
          //printf("%f  ", patch[j]);
        } //printf("\n\n");

        free(patch);
      }
      printf("\tFound %lld keypoints/patches of dimension %lld\n", numFeatures, ps*ps);
    }


    /**************** Deleting *************/
    vl_covdet_delete (covdet);
    free(fdata);
  }


  return  output  ;
}












/*

int size = 500;

if(get_image_width(image) > 500 && get_image_height(image) > 500){

  float* patchImage;
  VlCovDet * covdet = vl_covdet_new( method ) ;
  std::vector<double> patches ;
  std::vector<int> coords ;
  output[i].num_patches = 0;

  printf(" %d \n", (w/size)*(h/size));

  for(unsigned int a=0; a<w/size; a++){
    for(unsigned int b=0; b<h/size; b++){
      patchImage = (float*)vl_malloc(sizeof(float)*size*size);

      for(int p1 = 0; p1 < (signed)size; p1++){
        for(int p2 = 0; p2 < (signed)size; p2++){
          patchImage[ p1*size+p2 ]=fdata[ (a*size+p1)*w + b*(size)+p2 ];
        }
      }

      vl_covdet_put_image(covdet, patchImage, size, size) ;
      vl_covdet_detect(covdet) ;
      vl_covdet_drop_features_outside (covdet, boundaryMargin) ;
      vl_covdet_extract_affine_shape(covdet) ;
      vl_covdet_extract_orientations(covdet) ;
      vl_size numFeatures = vl_covdet_get_num_features(covdet) ;
      VlCovDetFeature const * feature = (const VlCovDetFeature*) vl_covdet_get_features(covdet) ;

      output[i].num_patches += numFeatures;
      output[i].dim_patches = ps*ps;
      printf("id %lld num %d  dim %d \n", (a*(w/size)+b), output[i].num_patches, output[i].dim_patches);


      for (unsigned int f = 0 ; f < numFeatures ; ++f) {
        float * patch = (float*)malloc(ps*ps*sizeof(float)) ;
        vl_covdet_extract_patch_for_frame(covdet, patch, patchResolution, patchRelativeExtent, patchRelativeSmoothing, feature[f].frame) ;

        //output[i].coords[2*f] = (int)round(feature[f].frame.x);
        //output[i].coords[2*f+1] = (int)round(feature[f].frame.y);
        //printf("x %f y %f \n", feature[i].frame.x, feature[i].frame.y);
        coords.push_back((int)round(feature[i].frame.x+size*a)); coords.push_back((int)round(feature[i].frame.y+size*b));
        if(f<10)  printf("\t\tid %d coords: %d %d\n", f,(int)round(feature[i].frame.x+size*a),(int)round(feature[i].frame.y+size*b));

        for (unsigned int j = 0 ; j < ps*ps ; ++j) {
          patches.push_back( patch[j] );
          //printf("%f  ", patch[j]);
        } //printf("\n\n");

        free(patch);
      }

      free(patchImage);
    }
  }

  printf(" %lld \n", output[i].num_patches*output[i].dim_patches*sizeof(double));
  output[i].coords = (int*) malloc(output[i].num_patches*2*sizeof(int));
  output[i].patches = (double*) malloc( output[i].num_patches*output[i].dim_patches*sizeof(double));
  for(int c=0; c<output[i].num_patches; c++){
    output[i].coords[c*2] = coords[2*c]; output[i].coords[c*2 + 1] = coords[2*c+1];
    for(int d=0; d<output[i].dim_patches; d++){
      output[i].patches[c*output[i].dim_patches + d] = patches[c*output[i].dim_patches + d];
    }
  }


*/
