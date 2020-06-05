
/**
			  * @file				lbp.cpp
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      LBP algorithm for texture descriptors computation.
			  */

#include "lbp.h"


descriptor_model_t lbp_features ( image_struct * image, unsigned int * seg )
{

  /*********** variables ******************/
  float *fdata;
  vl_size max_value = 0, min_value=10000000;
  int nsegs = 0;
  vl_size cellSize = 10;
  descriptor_model_t output;
  int w=get_image_width(image);
  int h=get_image_height(image);
  //int dim_sift_descriptor=128;
  int x,y;

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

    fdata = (float *) malloc(get_image_width(image)*get_image_height(image) * sizeof (float)) ;

    //rescale: rescales the values in the range [0.0, 255.0] as the algorithm is tuned for PGM images
    for(unsigned int j=0; j<get_image_width(image)*get_image_height(image);j++){
      fdata[j] = (float) 0 + ( (float)((get_image_data(image)[((i*get_image_width(image)*get_image_height(image)) + j)] - min_value)*(255-0)) / (float)(max_value-min_value) );
      //fdata[j] = (vl_sift_pix) get_image_data(image)[((i*get_image_width(image)*get_image_height(image)) + j)];
    }




    /****************** Make filter ******************/
    VlLbpMappingType mapping = VlLbpUniform;
		VlLbp* lbp = vl_lbp_new ( mapping, VL_FALSE);

		vl_size dimensionLbp = vl_lbp_get_dimension(lbp);

		long dim = floor(w/cellSize) * floor(h/cellSize) * dimensionLbp;
		float* features = (float *) malloc(dim * sizeof (float)) ;




    /****************** Process ******************/

		vl_lbp_process(	lbp,
				features,
				fdata,
				w,
				h,
				cellSize);

		for (int i = 0 ; i < floor(h/cellSize) ; i++) {
			for (int j = 0 ; j < floor(w/cellSize) ; j++) {

        Ds descriptor;
        int patchX = (int)floor(w/cellSize), patchY = (int)floor(h/cellSize);
				for (unsigned int b = 0 ; b < dimensionLbp ; b++) {
          double x = features[i*(int)floor(w/cellSize) + j*dimensionLbp + b];
          x = (double)min_value + (double)( ((x-0)*(double)(max_value-min_value))/(255-0) );
          descriptor.desc.push_back((float)x);
					printf("%f  ", x);
				}
        x = (patchX/2)+(patchX*j); y = (patchY/2)+(patchY*i);
				printf("[x: %d  y: %d]\n\n", x, y);
        printf("\n\n");

        output.descriptors[seg[y * get_image_width(image) + x]].push_back(descriptor);
			}
		}

    //printf("%lld %ld %lld %f\n", dimensionLbp, dim, cellSize, floor(h/cellSize));

    /***************** Finish up ******************/
    vl_lbp_delete(lbp);
		free(features);
  }

  /***************** Count descriptors per segment ******************/
  output.total_descriptors = 0;
  for(int ns=0;ns<output.num_segments;ns++){
    output.descriptors_per_segment[ns] = (int)output.descriptors[ns].size();
    output.total_descriptors += (int)output.descriptors[ns].size();
  }

  return  output  ;
}
