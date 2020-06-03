
/**
			  * @file				liop.cpp
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      LIOP algorithm for texture descriptors computation.
			  */

#include "liop.h"


descriptor_model_t liop_features ( image_struct * image, unsigned int * seg, detector_model_t * keypoints, float* parameters )
{

  /*********** variables ******************/
  vl_size w = get_image_width(image), h = get_image_height(image), max_value =0, min_value=10000000;
  float *fdata = 0 ;
  VlLiopDesc * liop ;
  int nsegs = 0;
  vl_size size = (vl_size) parameters[0] ; //11
  int number_neighbours = parameters[1]; //2
  int number_bins = parameters[2]; //5
  int radius = parameters[3]; //2
  float threshold = parameters[4]; //0.1
  descriptor_model_t output;
  vl_size dimension_desc=0;
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

  if((w%size == 0 && h%size != 0) || (w%size != 0 && h%size == 0) || (w%size != 0 && h%size != 0)){
    printf("\t* Assuming square image (croping)\n");
  } else {
    printf("\t* Square image (no croping needed)\n");
  }
  printf("\t* %d patch size \n\t  %d neighbours \n\t  %d bins \n\t  %d size of radius \n\t  %f intensity threshold\n", (int)size, (int)number_neighbours, (int)number_bins, (int)radius, threshold);

  for(unsigned int i=0;i<get_image_bands(image);i++){ //para cada dimensión
    printf("\n\n\t** Band %d **\n", i);

    /*********** rescaling: [0,255] ******************/
    find_maxmin(get_image_data(image)+(i*get_image_width(image)*get_image_height(image)),   get_image_width(image)*get_image_height(image),  &min_value,  &max_value);

    fdata = (float *) malloc(get_image_width(image)*get_image_height(image) * sizeof ((double) get_image_data(image)[0]) * sizeof (float)) ;

    //rescale: rescales the values in the range [0.0, 255.0] as the algorithm is tuned for PGM images
    for(unsigned int j=0; j<get_image_width(image)*get_image_height(image);j++){
      fdata[j] = (float) 0 + ( (float)((get_image_data(image)[((i*get_image_width(image)*get_image_height(image)) + j)] - min_value)*(255-0)) / (float)(max_value-min_value) );
      //fdata[j] = (float) get_image_data(image)[((i*get_image_width(image)*get_image_height(image)) + j)];
    }




    /****************** Make filter ******************/

    float * patch;
    liop = vl_liopdesc_new(number_neighbours, number_bins, radius, size);
    dimension_desc = vl_liopdesc_get_dimension(liop) ;
    //printf("\n\n\t-- %d --\n", dimension_desc);
    //printf("%d\n", dimension_desc);
    float * desc = (float*) vl_malloc(sizeof(float) * dimension_desc) ;
    vl_liopdesc_set_intensity_threshold(liop, threshold);




    /****************** Process each patch ******************/
    if(keypoints != NULL){
      patch=(float*)vl_malloc(sizeof(float)*keypoints[i].dim_patches);
      printf("\t %d descriptors\n", keypoints[i].num_patches);

      for(int p1=0; p1<keypoints[i].num_patches; p1++){
        for(int p2=0; p2<keypoints[i].dim_patches; p2++){
          patch[p2] = keypoints[i].patches[p1*keypoints[i].dim_patches+p2];
        }

        vl_liopdesc_process(liop, desc, patch) ;
        Ds descriptor;

        int x=keypoints[i].coords[2*p1],y=keypoints[i].coords[2*p1+1];
        //printf("coords: %f %f\n", x,y);

        for(unsigned int d=0; d<dimension_desc; d++){
          descriptor.desc.push_back((double) desc[d]);
          //printf("%f  ", desc[d]);
        }//printf("\n\n");

        output.descriptors[seg[y * get_image_width(image) + x]].push_back(descriptor);
      }
    }
    else{
      patch=(float*)vl_malloc(sizeof(float)*size*size);

      printf("\t %d descriptors\n", (int)((w/size)*(h/size)));
      for(unsigned int j=0; j<w/size; j++){
        for(unsigned int i=0; i<h/size; i++){
          for(int p1 = 0; p1 < (signed)size; p1++){
            for(int p2 = 0; p2 < (signed)size; p2++){
              patch[ p1*size+p2 ]=fdata[ (j*size+p1)*w + i*(size)+p2 ];
              //printf("%f  ", patch[p1*size+p2]);
            }//printf("\n");
          }//printf("\n\n");

          vl_liopdesc_process(liop, desc, patch) ;
          Ds descriptor;

          double x=j*size+(size/2),y=i*size+(size/2);
          //printf("coords: %f %f\n", x,y);

          for(unsigned int d=0; d<dimension_desc; d++){
            descriptor.desc.push_back((double) desc[d]);
            //printf("%f  ", desc[d]);
          }//printf("\n\n");

          output.descriptors[seg[(int)round(y) * get_image_width(image) + (int)round(x)]].push_back(descriptor);
        }
      }
    }


    /***************** Finish up ******************/
    vl_liopdesc_delete(liop) ;
		free(patch) ;
		free(desc) ;
  }




  /***************** Count descriptors per segment ******************/
  output.total_descriptors = 0;
  for(int ns=0;ns<output.num_segments;ns++){
    output.descriptors_per_segment[ns] = (int)output.descriptors[ns].size();
    output.total_descriptors += (int)output.descriptors[ns].size();
  }

  return  output  ;
}
