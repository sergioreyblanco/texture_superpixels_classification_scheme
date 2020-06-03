
/**
			  * @file				sift.cpp
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      SIFT algorithm for texture descriptors computation.
			  */

#include "sift.h"

float* flip_descriptor(float *src){
    float*dst = (float*)malloc(128*sizeof(float));
    int const BO = 8; /* number of orientation bins */
    int const BP = 4; /* number of spatial bins     */
    int i, j, t;

    for(j = 0; j < BP; ++j){
        int jp = BP - 1 - j;
        for(i = 0; i < BP; ++i){
            int o = BO*i + BP*BO*j;
            int op = BO*i + BP*BO*jp;
            dst[op] = src[o];
            for(t = 1; t < BO; ++t){
                dst[BO - t + op] = src[t+o];
            }
        }
    }

    return(dst);
}

descriptor_model_t raw_sift_features ( image_struct * image, unsigned int * seg, detector_model_t * keypoints, float* covdet_parameters )
{

  /*********** variables ******************/
  vl_sift_pix *fdata;
  vl_size max_value = 0, min_value=10000000;
  int nsegs = 0;
  descriptor_model_t output;
  int dim_sift_descriptor=128;
  vl_size patchResolution = (vl_size)covdet_parameters[1];
  vl_size patchSide = 2 * patchResolution + 1;
  double patchRelativeExtent = (double)covdet_parameters[2];
  double patchStep = (double)patchRelativeExtent / patchResolution;
  float * patch;
  float * patchXY;
  float tempDesc[128];

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
    find_maxmin(get_image_data(image)+(i*get_image_width(image)*get_image_height(image)),   get_image_width(image)*get_image_height(image),  &min_value,  &max_value);

    fdata = (vl_sift_pix *) malloc(get_image_width(image)*get_image_height(image) * sizeof ((double) get_image_data(image)[0]) * sizeof (vl_sift_pix)) ;

    //rescale: rescales the values in the range [0.0, 255.0] as the algorithm is tuned for PGM images
    for(unsigned int j=0; j<get_image_width(image)*get_image_height(image);j++){
      fdata[j] = (vl_sift_pix) 0 + ( (vl_sift_pix)((get_image_data(image)[((i*get_image_width(image)*get_image_height(image)) + j)] - min_value)*(255-0)) / (vl_sift_pix)(max_value-min_value) );
      //fdata[j] = (vl_sift_pix) get_image_data(image)[((i*get_image_width(image)*get_image_height(image)) + j)];
    }

    VlSiftFilt * sift = vl_sift_new(16, 16, 1, 3, 0);
    //vl_sift_set_magnif(sift, 3.0);

    patch=(float*)vl_malloc(sizeof(float)*keypoints[i].dim_patches);
    patchXY = (float*)malloc(2*sizeof(float)*patchSide*patchSide);
    printf("\t %d descriptors\n", keypoints[i].num_patches);


    for(int p1=0; p1<keypoints[i].num_patches; p1++){
      for(int p2=0; p2<keypoints[i].dim_patches; p2++){
        patch[p2] = keypoints[i].patches[p1*keypoints[i].dim_patches+p2];
      }

      vl_imgradient_polar_f(patchXY, patchXY+1,
                            2, 2 * patchSide,
                            patch, patchSide, patchSide, patchSide);

      vl_sift_calc_raw_descriptor(sift,
                                  patchXY,
                                  tempDesc,
                                  (int)patchSide, (int)patchSide,
                                  (double)(patchSide - 1) / 2, (double)(patchSide - 1) / 2,
                                  (double)patchRelativeExtent / (3.0 * (4 + 1) / 2) / patchStep,
                                  VL_PI / 2);

      float* desc = flip_descriptor(tempDesc);

      Ds descriptor;

      int x=keypoints[i].coords[2*p1],y=keypoints[i].coords[2*p1+1];
      //printf("coords: %f %f\n", x,y);

      for( int d=0; d<dim_sift_descriptor; d++){
        descriptor.desc.push_back((double) desc[d]);
        //printf("%f  ", desc[d]);
      }//printf("\n\n");

      output.descriptors[seg[y * get_image_width(image) + x]].push_back(descriptor);
    }


    /***************** Finish up ******************/
    free(patch);
    free(patchXY);
    vl_sift_delete(sift);
  }

  /***************** Count descriptors per segment ******************/
  output.total_descriptors = 0;
  for(int ns=0;ns<output.num_segments;ns++){
    output.descriptors_per_segment[ns] = (int)output.descriptors[ns].size();
    output.total_descriptors += (int)output.descriptors[ns].size();
  }

  printf("-- %d\n", output.total_descriptors);

  return  output  ;
}




descriptor_model_t sift_features ( image_struct * image, unsigned int * seg, float* thresholds )
{

  /*********** variables ******************/
  vl_sift_pix *fdata;
  VlSiftFilt *filt = 0 ;
  int octaves = -1, levels = 3, omin = -1 ;
  int* nkeys_len;
  int first = 1, err;
  VlSiftKeypoint const *keys = 0 ;
  int nkeys  ;
  VlSiftKeypoint const *k ;
  double angles [4] ;
  int nangles ;
  vl_size max_value = 0, min_value=10000000;
  int nsegs = 0;
  double edge_thresh  = thresholds[0] ;
  double peak_thresh  = thresholds[1] ;
  descriptor_model_t output;
  int dim_sift_descriptor=128;

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
    find_maxmin(get_image_data(image)+(i*get_image_width(image)*get_image_height(image)),   get_image_width(image)*get_image_height(image),  &min_value,  &max_value);

    fdata = (vl_sift_pix *) malloc(get_image_width(image)*get_image_height(image) * sizeof ((double) get_image_data(image)[0]) * sizeof (vl_sift_pix)) ;

    //rescale: rescales the values in the range [0.0, 255.0] as the algorithm is tuned for PGM images
    for(unsigned int j=0; j<get_image_width(image)*get_image_height(image);j++){
      fdata[j] = (vl_sift_pix) 0 + ( (vl_sift_pix)((get_image_data(image)[((i*get_image_width(image)*get_image_height(image)) + j)] - min_value)*(255-0)) / (vl_sift_pix)(max_value-min_value) );
      //fdata[j] = (vl_sift_pix) get_image_data(image)[((i*get_image_width(image)*get_image_height(image)) + j)];
    }




    /****************** Make filter ******************/
    filt = vl_sift_new (get_image_width(image), get_image_height(image), octaves, levels, omin) ;
    vl_sift_set_edge_thresh (filt, edge_thresh) ;
    vl_sift_set_peak_thresh (filt, peak_thresh) ;
    printf ("\toctaves = %d\n", vl_sift_get_noctaves(filt)) ;
    nkeys_len = (int *) malloc(vl_sift_get_noctaves(filt) * sizeof (int)) ;




    /****************** Process each octave ******************/
    while (1) {
      //calculate the GSS for the next octave
      if (first) {
        first = 0 ;
        err = vl_sift_process_first_octave (filt, fdata) ;
      } else {
        err = vl_sift_process_next_octave  (filt) ;
      }

      if (err) {
        err = VL_ERR_OK ;
        break ;
      }
      printf("\tGSS octave %d computed\n", vl_sift_get_octave_index (filt));


      // run detector
      vl_sift_detect (filt) ;

      keys  = vl_sift_get_keypoints (filt) ;
      nkeys = vl_sift_get_nkeypoints (filt) ;

      printf ("\tdetected %d (unoriented) keypoints\n", nkeys) ;
      nkeys_len[vl_sift_get_octave_index(filt) + 1] = nkeys;

      // for each keypoint
      for (int nk=0; nk < nkeys ; ++nk) {

        // obtain keypoint orientations
        k = keys + nk ;
        nangles = vl_sift_calc_keypoint_orientations(filt, angles, k) ;

        // for each orientation
        for (unsigned int q = 0 ; q < (unsigned) nangles ; ++q) {

          // compute descriptor
          Ds descriptor;
          vl_sift_pix desc[128];
          vl_sift_calc_keypoint_descriptor(filt, desc, k, angles [q]) ;


          // rescaling: rescales the values in the range of the HSI image
          for (int l = 0 ; l < dim_sift_descriptor ; ++l) {
            double x = (512.0 * desc[l]) ; //double x = (512.0 * descr [l]) ;
            //x = (x < 255.0) ? x : 255.0 ;

            //rescaling original range
            x = (double)min_value + (double)( ((x-0)*(double)(max_value-min_value))/(255-0) );

            descriptor.desc.push_back((vl_sift_pix)x);
          }

          output.descriptors[seg[(int)round(keys [nk].y) * get_image_width(image) + (int)round(keys [nk].x)]].push_back(descriptor);
        }
      }
    }




    /***************** Finish up ******************/
    //release filter
    if (filt) {
      vl_sift_delete (filt) ;
      filt = 0 ;
    }
    if (fdata) {
      free (fdata) ;
      fdata = 0 ;

      free (nkeys_len) ;
      nkeys_len = 0 ;
    }
    first = 1;
  }

  /***************** Count descriptors per segment ******************/
  output.total_descriptors = 0;
  for(int ns=0;ns<output.num_segments;ns++){
    output.descriptors_per_segment[ns] = (int)output.descriptors[ns].size();
    output.total_descriptors += (int)output.descriptors[ns].size();
  }

  return  output  ;
}
