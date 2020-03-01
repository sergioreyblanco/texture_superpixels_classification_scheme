
# Texture classification scheme


## Introduction

This is a classification scheme for multi and hyperspectral images. The process needs a multi or hyperspectral image (hsi), a ground truth (GT) training image and a GT testing image and goes as follows: in the training phase the labeled pixels (the ones different from zero) of the GT training image are taken from the hsi to create the predictive model and then in the testing phase the labeled pixels of the GT testing image are taken from the hsi and inserted into the predictive model to make a prediction. As output you receive a classification map (same width and height size of the input hsi and GTs) and a confusion matrix by class (one of the more useful metrics is the overall accuracy or OA). Labels in the GT start in 1 (0 labeled pixel means no GT information of that pixel) and finish in an integer indicating the number of clases of the classification problem.

It uses a SVM engine to perform the classification (training and testing phases) and has three work modes:

	1. Training and testing by pixels (using all the bands of the image as features).

	2. Training by pixels and testing by blocks (i.e. adjacent pixel sets).

	3. Training and testing using texture descriptors created with a selected texture pipeline (main work mode).

All the parameters are set using command line arguments and the scheme is designed to be used in the command line console.


## Installation and compilation

The scheme was tested under Ubuntu 17.04 and 18.04 systems. Follow these instructions to build it:

	1. Download the repository ``

	2. Install dependencies: `sudo apt-get install libblas-dev liblapacke-dev liblapack-dev libvlfeat-dev`.

	3. Compile: `make`.

	\*. Rebuild with `make clean; make`


## Execution and input arguments

The input arguments structure is the following one:

	./texture_classification_scheme [hsi] [train] [test] [options]

Without options and assuming you are using the well-known dataset *Salinas*, the command would be:

	./texture_classification_scheme ./data/Salinas.raw ./data/Salinas_train.raw ./data/Salinas_test.raw

However you can dive into the *advanced* mode (see next paragraph for description of all options):

	./texture_classification_scheme ./data/Salinas.raw ./data/Salinas_train.raw ./data/Salinas_test.raw -s data/Salinas_s10.raw -m output/cmap.ppm -f output/prediction.txt -t 1 -k 0 -c 0.01 -o output/output.model -v 1


## Citation
