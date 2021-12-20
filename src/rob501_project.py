from pathlib import Path
import argparse
import sys
import os
import cv2
from support.test_docker import test_docker
from support.panorama_producer import loading_images, stitch_panorama, produce_panorama

#----- ADD YOUR IMPORTS HERE IF NEEDED -----

def run_project(input_dir, output_dir):
    """
    Main entry point for your project code. 
    
    DO NOT MODIFY THE SIGNATURE OF THIS FUNCTION.
    """
    #---- FILL ME IN ----

    # Loading the images from the input directory.
    loaded_imgs = loading_images(input_dir)

    # Producing the panoramas and putting them in the output directory.
    produce_panorama(loaded_imgs, output_dir)

    #--------------------


# Command Line Arguments
parser = argparse.ArgumentParser(description='ROB501 Final Project.')
parser.add_argument('--input_dir', dest='input_dir', type=str, default="./input",
                    help='Input Directory that contains all required rover data')
parser.add_argument('--output_dir', dest='output_dir', type=str, default="./output",
                    help='Output directory where all outputs will be stored.')


if __name__ == "__main__":
    
    # Parse command line arguments
    args = parser.parse_args()

    # Uncomment this line if you wish to test your docker setup
    test_docker(Path(args.input_dir), Path(args.output_dir))

    # Run the project code
    run_project(args.input_dir, args.output_dir)