# DIP_Project

#### Dataset 
* The dataset is acquired from IAM handwritten Forms Dataset. It is downloaded from kaggle [https://www.kaggle.com/datasets/naderabdalghani/iam-handwritten-forms-dataset].
* The IAM handwritten forms dataset is created by more than 600 writers.
* We have taken that dataset, selected a subset of the dataset each image or paragraph pertaining to one unique writer to give the dataset its variability of writers. It contains different writing styles, cursive, non-cursive, with different line gaps and gaps between words. Hence providing variablity in the datset and making the segmentation approach and the overall pipeline more robust.
* The original dataset contains 1539 images and we have selected 30 of the images for out project.
* Each image contains 2 modalities, prited text and handwritten text. The handwritten paragraph text is cropped from the main image to get the desired image for our project. This is done for all the 30 images to form our handwritten dataset. 
