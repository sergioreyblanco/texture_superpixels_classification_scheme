
# Texture classification scheme


## Introduction

This is a classification scheme for multi and hyperspectral images. The process needs a multi or hyperspectral image (HSI), a ground truth (GT) training image and a GT testing image. It goes as follows: in the training phase the labeled pixels (the ones different from zero) of the GT training image are taken from the HSI to create the predictive model and then in the testing phase the labeled pixels of the GT testing image are taken from the HSI and inserted into the predictive model to make a prediction. As output you receive a classification map (same width and height size of the input HSI and GTs) and a confusion matrix (where one of the more useful metrics is the overall accuracy or OA). Labels in the GT start in 1 (0 labeled pixel means no GT information of that pixel) and finish in an integer indicating the total number of different clases in the classification problem.

It uses a SVM engine to perform the classification (training and testing phases) and has three work modes:

	1. Training and testing by pixels (using all the bands of the image as features).

	2. Training by pixels and testing by blocks (i.e. adjacent pixel sets).

	3. Training and testing using texture descriptors created with a selected texture pipeline (main work mode).

All the parameters are set using command line arguments and the scheme is designed to be used in the command line console.

The novelties of this classification scheme are the use of SLIC segmentation algorithm and texture algorithms. A descriptor vector (computed using the texture algorithms) and a label per segment is obtained to perform the classification process. The different texture techniques available are listed below. There are 27 different techniques which give different OA values.


## Installation and compilation

The scheme was tested under Ubuntu 17.04 and 18.04 systems. Follow these instructions to build it:

	1. Download the repository `https://github.com/sergioreyblanco/texture_superpixels_classification_scheme.git`.

	2. Install the dependencies: `apt-get install libblas-dev liblapacke-dev liblapack-dev libeigen3-dev libopenblas-dev`.

	3. Compile: `make`.

	\*. Rebuild with `make clean; make`


## Execution and input arguments

The input arguments structure are the following ones:

	./texture_classification_scheme [HSI] [train] [test] [options]

Without options and assuming you are using the remote sensing literature standard image *Salinas*, the command would be:

	./texture_classification_scheme ./data/Salinas.raw ./data/Salinas_train.raw ./data/Salinas_test.raw

However you can dive into the *advanced* mode (see next paragraph for description of all options):

	./texture_classification_scheme ./data/Salinas.raw ./data/Salinas_train.raw ./data/Salinas_test.raw -s ./data/Salinas_s10.raw -m ./output/cmap.ppm -f ./output/prediction.txt -t 1 -k 0 -c 0.01 -o ./output/output.model -v 1

All options:

	-s  -->  input_seg : input segmented image in RAW format | DEFAULT = segmentation algorithm applied to hyperspectral image
	-m  -->  output_clasfmap : output classification map | DEFAULT = output/map.ppm
	-f  -->  output_clasftxt : output classification textfile | DEFAULT = output/prediction.txt
	-p  -->  trainpredict_type : type of train and prediction procedure | DEFAULT = 3
			1 -- training by pixel and testing by pixel
			2 -- training by pixel and testing by blocks
			3 -- training by segments and testing by segments (most useful and fast mode)
	-k  -->  kernel_type : SVM kernel type | DEFAULT = 0
			0 -- LINEAR kernel
			1 -- POLYNOMIAL kernel
			2 -- RBF kernel
			3 -- SIGMOID kernel
	-c  -->  C : set the parameter C of C-SVC | DEFAULT = 0.02
	-o  -->  output_model : output SVM model generated in train phase | DEFAULT = output/output.model
	-v  -->  verbose : set the quiet or verbose mode | DEFAULT = false
	-t  -->  texture_pipeline : texture algorithms (pipeline to use) | DEFAULT = 0
			0  -- No texture algorithms
			1  -- Kmeans + VLAD
			2  -- Kmeans + BoW
			3  -- GMM + FisherVectors
			4  -- SIFT + Kmeans + VLAD
			5  -- SIFT + GMM + FisherVectors
			6  -- SIFT + Kmeans + VLAD (descriptors)
			7  -- SIFT + GMM + FisherVectors (descriptors)
			8  -- DSIFT + Kmeans + VLAD
			9  -- DSIFT + GMM + FisherVectors
			10 -- DSIFT + Kmeans + VLAD (descriptors)
			11 -- DSIFT + GMM + FisherVectors (descriptors)
			12 -- LIOP + Kmeans + VLAD
			13 -- LIOP + GMM + FisherVectors
			14 -- LIOP + Kmeans + VLAD (descriptors)
			15 -- LIOP + GMM + FisherVectors (descriptors)
			16 -- HOG + Kmeans + VLAD
			17 -- HOG + GMM + FisherVectors
			18 -- HOG + Kmeans + VLAD (descriptors)
			19 -- HOG + GMM + FisherVectors (descriptors)
			20 -- LBP + Kmeans + VLAD
			21 -- LBP + GMM + FisherVectors
			22 -- LBP + Kmeans + VLAD (descriptors)
			23 -- LBP + GMM + FisherVectors (descriptors)
			24 -- MSER&SIFT + Kmeans + VLAD
			25 -- MSER&SIFT + GMM + FisherVectors
			26 -- MSER&SIFT + Kmeans + VLAD (descriptors)
			27 -- MSER&SIFT + GMM + FisherVectors (descriptors)
	-4  -->  sift_thresholds : 2 thresholds for the SIFT algorithm | DEFAULT = 0.1 2.5 | Parameter description:
			* peak_threshold : filters peaks of the DoG scale space that are too small (in absolute value).
			* edge_threshold : eliminates peaks of the DoG scale space whose curvature is too small.
	-7  -->  dsift_parameters : 4 parameters for the DSIFT algorithm | DEFAULT = 2 4 4 8
			* bin_size : size of the spatial bins in the X and Y planes.
			* num_bin_X : number of spatial bins in the X plane.
			* num_bin_Y : number of spatial bins in the Y plane.
			* num_bin_T : number of orientation bins.
	-12 -->  LIOP_parameters : 5 parameters for the LIOP algorithm | DEFAULT = 11 2 5 2 0.1
			* side_length : width of the input image patch (the patch is square).
			* num_neighbours	: number of neighbours.
			* num_spatial_bins : number of spatial bins.
			* radius : radius of the circular sample neighbourhoods.
			* intensity_threshold : non-negative, the threshold as is is used when comparing intensities; negative, the absolute value of the specified number is multipled by the maximum intensity difference inside a patch to obtain the threshold.
	-5  -->  HOG_parameters : 3 parameters for the HOG algorithm | DEFAULT = 32 8 0
	    * num_orientations	: number of distinguished orientations.
			* cell_size : size of a HOG cell.
			* bilinear_orientation_assignments : 1 if orientations should be assigned with bilinear interpolation.
	-6  -->  LBP_parameters : 1 parameter for the LBP algorithm | DEFAULT = 100
			* cell_size : size of the LBP cells.
	-r  -->  reduction_method : mean or PCA reduction methods | DEFAULT = 1
			1 -- mean
			3 -- PCA

	* Parameter values -p (1 or 2) and -t (any) are mutually exclusive. So the situations "-p 1 -t (any)" and "-p 2 -t (any)" are forbidden.



## Citation

Rey, S. R., Blanco, D. B., & Argüello, F. (2020). Texture Extraction Techniques for the Classification of Vegetation Species in Hyperspectral Imagery: Bag of Words Approach Based on Superpixels. Remote Sensing, 12(16), 2633.
