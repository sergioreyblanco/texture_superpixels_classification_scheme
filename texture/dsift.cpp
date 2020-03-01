
/**
			  * @file				dsift.cpp
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      DSIFT algorithm for texture descriptors computation.
			  */

#include "dsift.h"

//finds max and min values
void find_maxmin_dsift(unsigned int *data, int numData, vl_size* min_value, vl_size* max_value)
{
  vl_size ma = 0, mi=10000000;

  for(int i=0; i<numData;i++){
    if(data[i] > ma){
      ma = data[i];
    } else if(data[i] < mi){
      mi = data[i];
    }
  }

  (*max_value) = ma;
  (*min_value) = mi;
}


sift_model_t dsift_features ( image_struct * image, unsigned int * seg, int* parameters )
 {

    /*********** variables ******************/
    //dim_sift_descriptor=dim;
    vl_size max_value =0, min_value=10000000;
    VlDsiftDescriptorGeometry geom ;
    VlDsiftKeypoint const *frames ;
    VlDsiftFilter *dsift ;
    vl_bool useFlatWindow = VL_FALSE ;
    int numFrames ;
    int descrSize ;
    int stepX ;
    int stepY ;
    int minX ;
    int minY ;
    int maxX ;
    int maxY ;
    float const *descrs ;
    int nsegs = 0;
    Ds descriptor;
    float *fdata;
    sift_model_t output;

    for(unsigned int sg=0;sg < get_image_width(image)*get_image_height(image);sg++){
      if(seg[sg] > (unsigned int)nsegs){
        nsegs = seg[sg];
      }
    }

    output.num_segments = nsegs+1;
    output.descriptors = new std::vector<Ds>[output.num_segments];
    output.descriptors_per_segment = new int[output.num_segments];

    for(unsigned int i=0;i<get_image_bands(image);i++){ //para cada dimensión
      printf("\n\n\t** Band %d **\n", i);


      /*********** rescaling: [0,255] ******************/
      find_maxmin_dsift(get_image_data(image)+(i*get_image_width(image)*get_image_height(image)),   get_image_width(image)*get_image_height(image),  &min_value,  &max_value);

      fdata = (float *) malloc(get_image_width(image)*get_image_height(image) * sizeof ((float) get_image_data(image)[0]) * sizeof (float)) ;

      //rescale: rescales the values in the range [0.0, 255.0] as the algorithm is tuned for PGM images
      for(unsigned int j=0; j<get_image_width(image)*get_image_height(image);j++){
        fdata[j] = (float) 0 + ( ((get_image_data(image)[((i*get_image_width(image)*get_image_height(image)) + j)] - min_value)*(255-0)) / (max_value-min_value) );
      }




      /*********** DSIFT creation and params ******************/
      dsift = vl_dsift_new (get_image_width(image),get_image_height(image)) ; //width, height

      //4,4,4 y 3; 2,2,4 y 1; 4,4,2 y 1;
      geom.numBinX = parameters[1] ; geom.numBinY = parameters[2] ; geom.numBinT = parameters[3] ; //mas grande
      geom.binSizeX = parameters[0] ; geom.binSizeY = parameters[0] ;//mas pequeño
      vl_dsift_set_geometry(dsift, &geom) ;

      vl_dsift_set_steps(dsift, parameters[0], parameters[0]) ;//mas pequeño

      vl_dsift_set_bounds(dsift,
                  0,
                  0,
                  get_image_width(image)-1,
                  get_image_height(image)-1);

      vl_dsift_set_flat_window(dsift, 0) ;//probar con 1

      vl_dsift_set_window_size(dsift, 1) ;//aumentar o disminuir esto





      /*********** DSIFT printing params ******************/

      numFrames = vl_dsift_get_keypoint_num (dsift) ;
      printf("\tvl_dsift: num of features:   %d\n", numFrames );

      descrSize = vl_dsift_get_descriptor_size (dsift) ;
      printf("\tvl_dsift: descriptor size:   %d\n", descrSize);

      geom = *vl_dsift_get_geometry (dsift) ;
      printf("\tvl_dsift: num bins:          [numBinT, numBinX, numBinY] = [%d, %d, %d]\n",
                    geom.numBinT,
                    geom.numBinX,
                    geom.numBinY);
      printf("\tvl_dsift: bin sizes:         [binSizeX, binSizeY] = [%d, %d]\n",
                    geom.binSizeX,
                    geom.binSizeY);

      vl_dsift_get_steps (dsift, &stepY, &stepX) ;
      printf("\tvl_dsift: subsampling steps: stepX=%d, stepY=%d\n", stepX, stepY);

      vl_dsift_get_bounds (dsift, &minY, &minX, &maxY, &maxX) ;
      printf("\tvl_dsift: bounds:            [minX,minY,maxX,maxY] = [%d, %d, %d, %d]\n",
                    minX+1, minY+1, maxX+1, maxY+1);

      useFlatWindow = vl_dsift_get_flat_window(dsift) ;
      printf("\tvl_dsift: flat window:       %s\n", VL_YESNO(useFlatWindow));
      printf("\tvl_dsift: window size:       %g\n", vl_dsift_get_window_size(dsift)) ;





      /*********** processing ******************/
      vl_dsift_process (dsift, fdata) ;

      frames = vl_dsift_get_keypoints (dsift) ;
      descrs = vl_dsift_get_descriptors (dsift) ;


      float *tmpDescr = (float*)malloc(sizeof(float) * descrSize) ;

      for (int k = 0 ; k < numFrames ; ++k) {
        //printf("keypoint coord: %f %f \n", frames[k].x, frames[k].y) ;


        vl_dsift_transpose_descriptor (tmpDescr,
                     descrs + descrSize * k,
                     geom.numBinT,
                     geom.numBinX,
                     geom.numBinY) ;
        for (int i = 0 ; i < descrSize ; ++i) {
            short x = (512.0 * tmpDescr[i]) ;
            x = min_value + ( ((x-0)*(max_value-min_value))/(255-0) );
            descriptor.desc[i] = x;
        }
        output.descriptors[seg[(int)frames[k].y * get_image_width(image) + (int)frames[k].x]].push_back(descriptor);
      }



      /***************** Count descriptors per segment ******************/
      output.total_descriptors = 0;
      for(int ns=0;ns<output.num_segments;ns++){
        output.descriptors_per_segment[ns] = output.descriptors[ns].size();
        output.total_descriptors += output.descriptors[ns].size();
      }



      /*********** liberation and exiting ******************/
      free(tmpDescr) ;
      vl_dsift_delete (dsift) ;

      /*for (int i = 0 ; i < output.num_segments ; ++i) {
        for (int j = 0 ; j < output.descriptors[i].size(); ++j) {
          for (int k = 0 ; k < 128; ++k) {
            printf("%f  ", output.descriptors[i][j].desc[k]);
          }
          printf("\n");
        }
      }
      printf("\n\n\n");*/
  }

  return  output  ;
}


sift_model_t dsift_basic_features ( image_struct * image, unsigned int * seg )
{

  /*********** variables ******************/
  vl_size max_value =0, min_value=10000000;
  VlDsiftDescriptorGeometry geom ;
  VlDsiftKeypoint const *frames ;
  int numFrames ;
  int descrSize ;
  float const *descrs ;
  int nsegs = 0;
  Ds descriptor;
  float *fdata;
  sift_model_t output;

  for(unsigned int sg=0;sg < get_image_width(image)*get_image_height(image);sg++){
    if(seg[sg] > (unsigned int)nsegs){
      nsegs = seg[sg];
    }
  }

  output.num_segments = nsegs+1;
  output.descriptors = new std::vector<Ds>[output.num_segments];
  output.descriptors_per_segment = new int[output.num_segments];

  for(unsigned int i=0;i<get_image_bands(image);i++){ //para cada dimensión
    printf("\n\n\t** Band %d **\n", i);


    /*********** rescaling: [0,255] ******************/
    find_maxmin_dsift(get_image_data(image)+(i*get_image_width(image)*get_image_height(image)),   get_image_width(image)*get_image_height(image),  &min_value,  &max_value);

    fdata = (float *) malloc(get_image_width(image)*get_image_height(image) * sizeof ((float) get_image_data(image)[0]) * sizeof (float)) ;

    //rescale: rescales the values in the range [0.0, 255.0] as the algorithm is tuned for PGM images
    for(unsigned int j=0; j<get_image_width(image)*get_image_height(image);j++){
      fdata[j] = (float) 0 + ( ((get_image_data(image)[((i*get_image_width(image)*get_image_height(image)) + j)] - min_value)*(255-0)) / (max_value-min_value) );
    }




    /*********** DSIFT creation and params ******************/
    VlDsiftFilter* vlf = vl_dsift_new_basic(get_image_width(image),get_image_height(image), 4, 8);

    /*********** DSIFT printing params ******************/

    numFrames = vl_dsift_get_keypoint_num (vlf) ;
    printf("\tvl_dsift: num of features:   %d\n", numFrames );

    descrSize = vl_dsift_get_descriptor_size (vlf) ;
    printf("\tvl_dsift: descriptor size:   %d\n", descrSize);

    geom = *vl_dsift_get_geometry (vlf) ;
    printf("\tvl_dsift: num bins:          [numBinT, numBinX, numBinY] = [%d, %d, %d]\n",
                  geom.numBinT,
                  geom.numBinX,
                  geom.numBinY);
    printf("\tvl_dsift: bin sizes:         [binSizeX, binSizeY] = [%d, %d]\n",
                  geom.binSizeX,
                  geom.binSizeY);


    /*********** processing ******************/
    vl_dsift_process (vlf, fdata) ;

    int Nkeypoints = vl_dsift_get_keypoint_num(vlf);
    int descSize = vl_dsift_get_descriptor_size(vlf);
    frames = vl_dsift_get_keypoints (vlf) ;
    descrs = vl_dsift_get_descriptors (vlf) ;

    float *tmpDescr = (float*)malloc(sizeof(float) * descSize) ;


    for (int k = 0 ; k < Nkeypoints ; ++k) {
      //printf("keypoint coord: %f %f \n", frames[k].x, frames[k].y) ;

      vl_dsift_transpose_descriptor (tmpDescr,
                   descrs + descSize * k,
                   geom.numBinT,
                   geom.numBinX,
                   geom.numBinY) ;
      for (int i = 0 ; i < descSize ; ++i) {
          double x = (512.0 * tmpDescr[i]) ;
          x = min_value + ( ((x-0)*(max_value-min_value))/(255-0) );
          descriptor.desc[i] = x;
      }
      output.descriptors[seg[(int)frames[k].y * get_image_width(image) + (int)frames[k].x]].push_back(descriptor);
    }




    /***************** Count descriptors per segment ******************/
    output.total_descriptors = 0;
    for(int ns=0;ns<output.num_segments;ns++){
      output.descriptors_per_segment[ns] = output.descriptors[ns].size();
      output.total_descriptors += output.descriptors[ns].size();
    }




    /*********** liberation and exiting ******************/
    vl_dsift_delete (vlf) ;

    /*for (int i = 0 ; i < output.num_segments ; ++i) {
      for (int j = 0 ; j < output.descriptors[i].size(); ++j) {
        for (int k = 0 ; k < 128; ++k) {
          printf("%f  ", output.descriptors[i][j].desc[k]);
        }
        printf("\n");
      }
    }
    printf("\n\n\n");*/
  }

  return  output  ;
}
