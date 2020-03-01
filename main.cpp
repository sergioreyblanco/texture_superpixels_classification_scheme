
/**
			  * @file				main.cpp
			  *
				* @author			Sergio Rey Blanco
				*
			  * @brief      Classification scheme using a texture descriptors and a SVM engine.
			  */


#include "texture/slic.h"
#include "texture/texture_pipelines.h"
#include "svm/trainpredict.h"
#include "utility/data_structures.h"
#include "utility/load_data.h"
#include "utility/general_utilities.h"



int main(int argc, char **argv){




  /************ Important variables **********/

  struct svm_parameter svm_param;		// set by parse_command_line
  struct svm_problem svm_prob;		// set by read_problem
  struct svm_model *svm_model;  // SVM model created by the training phase and used in the predicting phase
  struct svm_node *X_matrix = NULL; // data matrix with the prediction variables
  bool cross_validation = false; // whether to do or not CV
  int nr_fold = 5; // number of folds of the CV
  image_struct *image, *train_image; // hyperspectral data structures
  reference_data_struct *gt_train, *gt_test, *gt_train_image; // reference data structures
  segmentation_struct *seg_image = NULL; // segmentation data structure
  command_arguments_struct* command_arguments; // input command arguments data structure
  texture_struct* descriptors = NULL, *descriptors_train; // texture descriptors data structures
  char error[1500], message[1500]; //strings for info and error printing





  /************************  Train  ************************/

  // parsing the input command arguments
  command_arguments = (command_arguments_struct*)malloc(sizeof(command_arguments_struct));
	parse_command_line(argc, argv, command_arguments, &svm_param, error);

  // loading the hyperspectral image into memory
  image=(image_struct*)malloc(sizeof(image_struct));
  load_hsi(image, command_arguments->input_hsi, error);

  // loading the training reference data image into memory
  gt_train=(reference_data_struct*)malloc(sizeof(reference_data_struct));
  load_gt(gt_train, command_arguments->input_gttrain, "train", error);

  // loading the testing reference data image into memory
  gt_test=(reference_data_struct*)malloc(sizeof(reference_data_struct));
  load_gt(gt_test, command_arguments->input_gttest, "test", error);

  // removing the unlabeled pixels from the hyperspectral image
  train_image = (image_struct*)malloc(sizeof(image_struct));
  remove_unlabeled_hsi(image, gt_train, train_image);

  // removing the unlabeled pixels from the reference data image
  gt_train_image = (reference_data_struct*)malloc(sizeof(reference_data_struct));
  remove_unlabeled_gt(gt_train, gt_train_image);

  // tasks only done if texture descriptors needed
  if(command_arguments->trainpredict_type == 3){
    seg_image=(segmentation_struct*)malloc(sizeof(segmentation_struct));

    // if not segmented image introduced
    if(command_arguments->input_seg[0] == -1){

      start_crono("SLIC");

      // segmentation algorithm over the hyperspectral image
      do_segmentation(1, image, seg_image, error);

      stop_crono();

    } else{

      // loading a previously done segmentated image
      load_segmentation(seg_image, command_arguments->input_seg, error);
    }

    // texture descriptors calculation
    descriptors = texture_pipeline(image, seg_image, gt_train, get_reference_data_width(gt_train_image), command_arguments, error);

    // removing the unlabeled descriptors form the descriptors data structure
    descriptors_train = (texture_struct*)malloc(sizeof(texture_struct));
    remove_unlabeled_descriptors(descriptors, descriptors_train);
  }


  //load_problem_txt(input_file_name_train, X_matrix);
  // problem loading for pixel or block training and predicting
  if(command_arguments->trainpredict_type == 1 || command_arguments->trainpredict_type == 2){
    load_problem_hsi(train_image, gt_train_image, get_reference_data_width(gt_train_image), &svm_prob, X_matrix);
  }
  // problem loading for texture descriptors training and predicting
  else if(command_arguments->trainpredict_type == 3){
    load_problem_texture(descriptors_train, &svm_prob, X_matrix);
  }


  // need task before SVM training
	if(const char* e = svm_check_parameter(&svm_prob,&svm_param)) {
    sprintf(error, "%s", e);
    print_error((char*)error);
		return(EXIT_FAILURE);
	}

  // if cross validation is needed
	if(cross_validation) {
		do_cross_validation(nr_fold, svm_param, svm_prob);
	}
  // simple training (without CV)
	else {
    start_crono("TRAIN");

		svm_model = svm_train(&svm_prob,&svm_param);

    stop_crono();

    // saving of the created model into disk
    if(command_arguments->output_model[0] != -1) {
  		if(svm_save_model(command_arguments->output_model, svm_model)) {
        sprintf(error, "Cannot save model to file %s", command_arguments->output_model);
        print_error((char*)error);
  			exit(1);
  		} else {
        sprintf(message, "Saved " UNDERLINED "SVM model" RESET GREEN " : %s", command_arguments->output_model);
        print_info((char*)message);
      }
    }
	}





  /************************  Predict  ************************/

  // reading a previously created model from disk
  if(command_arguments->output_model[0] != -1){
    if((svm_model=svm_load_model(command_arguments->output_model))==0) {
      sprintf(error, "Cannot open model from file %s", command_arguments->output_model);
      print_error((char*)error);
      exit(1);
  	} else {
      sprintf(message, "Loaded " UNDERLINED "SVM model" RESET GREEN " : %s", command_arguments->output_model);
      print_info((char*)message);
    }
  }

  // type of prediction checking
	if(svm_param.probability) {
		if(svm_check_probability_model(svm_model)==0) {
      sprintf(error, "Model does not support probabiliy estimates");
      print_error((char*)error);
			exit(1);
		}
	} else {
		if(svm_check_probability_model(svm_model)!=0) {
      sprintf(message, "Model supports probability estimates, but disabled in prediction.\n");
      print_info((char*)message);
    }
	}

  start_crono("TEST");

  //predict_txt(input_test,output_test);
  // prediction for pixel or block training and predicting
	if(command_arguments->trainpredict_type == 1 || command_arguments->trainpredict_type == 2){
    predict_hsi(command_arguments, svm_param, svm_model, image, gt_test, error, message);
  }
  // prediction for texture descriptors training and predicting
  else if(command_arguments->trainpredict_type == 3){
    predict_texture(command_arguments, descriptors, svm_model, seg_image, gt_test, error, message);
  }

  stop_crono();





  /************************  Resources liberation  ************************/

  svm_free_and_destroy_model(&svm_model);
  svm_destroy_param(&svm_param);
  free(svm_prob.y);
  free(svm_prob.x);
  free(X_matrix);


	return (EXIT_SUCCESS);
}
