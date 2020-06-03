
/**
			  * @file				hog.cpp
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      HOG algorithm for texture descriptors computation.
			  */

#include "hog.h"



descriptor_model_t hog_features ( image_struct * image, unsigned int * seg, int* parameters )
{


  /*********** variables ******************/
  vl_size max_value =0, min_value=10000000;
  float *fdata = 0 ;
  int nsegs = 0;
  vl_size  numOrientations = parameters[0], cellSize = parameters[1]; vl_bool bilinearOrientationAssignments = parameters[2];
  vl_size hogWidth, hogHeight, hogDimension;
  float* hogArray;
  descriptor_model_t output;
  VlHogVariant  variant=(VlHogVariant)0;
  VlHog * hog;
  vl_size w=get_image_width(image), h=get_image_height(image), dimension = get_image_bands(image);
  int x, y;
  VlRand rand ;

  vl_rand_init (&rand) ;
	vl_rand_seed (&rand,  1000) ;

  for(unsigned int sg=0;sg < get_image_width(image)*get_image_height(image);sg++){
    if(seg[sg] > (unsigned int)nsegs){
      nsegs = seg[sg];
    }
  }

  output.num_segments = nsegs+1;
  output.descriptors = new std::vector<Ds>[output.num_segments];
  output.descriptors_per_segment = new int[output.num_segments];


  find_maxmin(get_image_data(image), get_image_width(image)*get_image_height(image)*get_image_bands(image),  &min_value,  &max_value);

  fdata = (float *) malloc(get_image_width(image)*get_image_height(image)*get_image_bands(image) * sizeof (float)) ;

  for(unsigned int j=0; j<get_image_width(image)*get_image_height(image)*get_image_bands(image);j++){
    fdata[j] = (float) 0 + ( (float)((get_image_data(image)[j] - min_value)*(255-0)) / (float)(max_value-min_value) );
  }


	/*********** process ******************/

  printf("\t* %d orientations \n\t  %d cellSize \n\t  %d bilinearOrientationAssignments \n", (int)numOrientations, (int)cellSize, (int)bilinearOrientationAssignments);

	hog = vl_hog_new(variant, numOrientations, VL_FALSE) ;
	vl_hog_set_use_bilinear_orientation_assignments ( hog, bilinearOrientationAssignments);
	vl_hog_put_image(hog, fdata, w, h, dimension, cellSize) ;
	hogWidth = vl_hog_get_width(hog) ;
	hogHeight = vl_hog_get_height(hog) ;
	hogDimension = vl_hog_get_dimension(hog) ;
	hogArray = (float*)malloc(hogWidth*hogHeight*hogDimension*sizeof(float)) ;
	vl_hog_extract(hog, hogArray) ;
	//vl_index const* perm;
	//perm=vl_hog_get_permutation( hog );



  int patchX = (int)round(w/hogWidth), patchY = (int)round(h/hogHeight);
  for (unsigned int i = 0 ; i < hogHeight ; i++) {
    for (unsigned int j = 0 ; j < hogWidth ; j++) {

      Ds descriptor;
      for (unsigned int b = 0 ; b < hogDimension ; b++) {
        //printf("%f  ", hogArray[i*j*hogDimension+b]);
        if(hogArray[i*j*hogDimension+b] != hogArray[i*j*hogDimension+b]){
          printf("\n\n\n**\n");
          hogArray[i*j*hogDimension+b]=0.0;
        }
        descriptor.desc.push_back((double) hogArray[i*j*hogDimension+b]);
      }
      x = (int)round((patchX/2)+(patchX*j)); y = (int)round((patchY/2)+(patchY*i));
      output.descriptors[seg[y * get_image_width(image) + x]].push_back(descriptor);
      //printf("[x: %d  y: %d]\n\n", x, y);
    }
  }

  printf("\t* hogWidth: %lld, hogHeight: %lld, hogDimension: %lld\n", hogWidth, hogHeight, hogDimension);



	/*********** delete ******************/
	vl_hog_delete(hog) ;
	free(hogArray);




  /***************** Count descriptors per segment ******************/
  output.total_descriptors = 0;
  for(int ns=0;ns<output.num_segments;ns++){
    output.descriptors_per_segment[ns] = (int)output.descriptors[ns].size();
    output.total_descriptors += (int)output.descriptors[ns].size();
  }

  return  output  ;
}
