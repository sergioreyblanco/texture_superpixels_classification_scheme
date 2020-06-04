
# Texture classification scheme


## Introduction

This is a classification scheme for multi and hyperspectral images. The process needs a multi or hyperspectral image (hsi), a ground truth (GT) training image and a GT testing image and goes as follow: in the training phase the labeled pixels (the ones different from zero) of the GT training image are taken from the hsi to create the predictive model and then in the testing phase the labeled pixels of the GT testing image are taken from the hsi and inserted into the predictive model to make a prediction. As output you receive a classification map (same width and height size of the input hsi and GTs) and a confusion matrix by class (one of the more useful metrics is the overall accuracy or OA). Labels in the GT start in 1 (0 labeled pixel means no GT information of that pixel) and finish in an integer indicating the number of clases of the classification problem.

It uses a SVM engine to perform the classification (training and testing phases) and has three work modes:

	1. Training and testing by pixels (using all the bands of the image as features).

	2. Training by pixels and testing by blocks (i.e. adjacent pixel sets).

	3. Training and testing using texture descriptors created with a selected texture pipeline (main work mode).

All the parameters are set using command line arguments and the scheme is designed to be used in the command line console.

The novelties of this classification scheme are the use of SLIC segmentation algorithm and other texture techniques. A descriptor vector and label per segment is obtained to perform the classification process.


## Installation and compilation

The scheme was tested under Ubuntu 17.04 and 18.04 systems. Follow these instructions to build it:

	1. Download the repository ``

	2. Install dependencies: `apt-get install libblas-dev liblapacke-dev liblapack-dev libeigen3-dev libopenblas-dev`.

	3. Compile: `make`.

	\*. Rebuild with `make clean; make`


## Execution and input arguments

The input arguments structure are the following ones:

	./texture_classification_scheme [hsi] [train] [test] [options]

Without options and assuming you are using the remote sensing literature standard image *Salinas*, the command would be:

	./texture_classification_scheme ./data/Salinas.raw ./data/Salinas_train.raw ./data/Salinas_test.raw

However you can dive into the *advanced* mode (see next paragraph for description of all options):

	./texture_classification_scheme ./data/Salinas.raw ./data/Salinas_train.raw ./data/Salinas_test.raw -s ./data/Salinas_s10.raw -m ./output/cmap.ppm -f ./output/prediction.txt -t 1 -k 0 -c 0.01 -o ./output/output.model -v 1

All options:

	-s  -->  input_seg : input segmented image in RAW format | DEFAULT = segmentation algorithm applied to hyperspectral image
	-m  -->  output_clasfmap : output classification map | DEFAULT = ouput/map.ppm
	-f  -->  output_clasftxt : output classification textfile | DEFAULT = ouput/prediction.txt
	-p  -->  trainpredict_type : type of train and prediction procedure | DEFAULT = 3
			1 -- by pixel
			2 -- by blocks
			3 -- by segments
	-k  -->  kernel_type : SVM kernel type | DEFAULT = 0
			0 -- LINEAR kernel
			1 -- POLYNOMIAL kernel
			2 -- RBF kernel
			3 -- SIGMOID kernel
	-c  -->  C : set the parameter C of C-SVC | DEFAULT = 0.02
	-o  -->  output_model : output SVM model generated in train phase | DEFAULT = output/output.model
	-v  -->  verbose : set the quiet or verbose mode | DEFAULT = true
	-t  -->  texture_pipeline : texture algorithms (pipeline to use) | DEFAULT = 0
			0 -- no textures
			1 -- kmeans + vlad
			2 -- kmeans + bow
			3 -- gmm + fishervectors
			4 -- sift + km + vlad
			5 -- sift + gmm + fishervectors
			6 -- sift
			7 -- dsift + km + vlad
			8 -- dsift + gmm + fishervectors
			9 -- dsift
	-4  -->  sift_thresholds : 2 thresholds for the SIFT algorithm | DEFAULT = 0.1 2.5
	-7  -->  dsift_parameters : 4 parameters for the DSIFT algorithm | DEFAULT = 2 4 4 8
	-12  -->  LIOP_parameters : 5 parameters for the LIOP algorithm | DEFAULT = 11 2 5 2 0.1
	-5  -->  HOG_parameters : 4 parameters for the HOG algorithm | DEFAULT = 32 8 FALSE
	-r  -->  reduction_method : mean, PCA, ... reduction methods | DEFAULT = 1

	* Parameters -t (1 or 2) and -p (any) are mutually exclusive;



## Citation
