# Panoramic-Stitching-with-Feature-Detection

## About this Project

In this project, we demonstrate that feature detection and homography techniques such as RANSAC, SIFT, and DLT can create a non-segmenting and non-vignetting panormama. 
This project uses 5 images from run 3 and run 5 of the Canadian Planetary Emulation Terrain Energy-Aware Rover Navigation Dataset. In the project, we find that these techniques can detect all the necessary key features and warp the images together.

### Dependencies

Numpy: https://numpy.org/doc/stable/index.html#

OpenCV: https://opencv.org/releases/

## Getting Started

Unfortuntately, it can't be run because the input folder is too big (>20GB) to be put on here. As of right now, you can get the data from https://starslab.ca/enav-planetary-dataset/. 

However, once you get the dataset, you should be able to run the code. You need to rename the data folder as `input` and put it in the same directory as the `output` folder. The directory structure should be in the `/src/support/panorama_producer.py` file.
