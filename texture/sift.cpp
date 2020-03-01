
/**
			  * @file				sift.cpp
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      SIFT algorithm for texture descriptors computation.
			  */

#include "sift.h"

//finds max and min values
template < typename D >
void find_maxmin(D *data, int numData, vl_size* min_value, vl_size* max_value)
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


sift_model_t sift_features ( image_struct * image, unsigned int * seg, float* thresholds )
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
  Ds descriptor;
  double edge_thresh  = thresholds[0] ;
  double peak_thresh  = thresholds[1] ;
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
    find_maxmin(get_image_data(image)+(i*get_image_width(image)*get_image_height(image)),   get_image_width(image)*get_image_height(image),  &min_value,  &max_value);

    fdata = (vl_sift_pix *) malloc(get_image_width(image)*get_image_height(image) * sizeof ((double) get_image_data(image)[0]) * sizeof (vl_sift_pix)) ;

    //rescale: rescales the values in the range [0.0, 255.0] as the algorithm is tuned for PGM images
    for(unsigned int j=0; j<get_image_width(image)*get_image_height(image);j++){
      fdata[j] = (vl_sift_pix) 0 + ( (vl_sift_pix)((get_image_data(image)[((i*get_image_width(image)*get_image_height(image)) + j)] - min_value)*(255-0)) / (vl_sift_pix)(max_value-min_value) );
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
          vl_sift_calc_keypoint_descriptor(filt, descriptor.desc, k, angles [q]) ;

          // rescaling: rescales the values in the range of the HSI image
          for (int l = 0 ; l < dim_sift_descriptor ; ++l) {
            double x = (512.0 * descriptor.desc[l]) ; //double x = (512.0 * descr [l]) ;
            //x = (x < 255.0) ? x : 255.0 ;

            //rescaling original range
            x = (double)min_value + (double)( ((x-0)*(double)(max_value-min_value))/(255-0) );

            descriptor.desc[l] = (vl_sift_pix)x;
          }

          output.descriptors[seg[(int)keys [nk].y * get_image_width(image) + (int)keys [nk].x]].push_back(descriptor);
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
