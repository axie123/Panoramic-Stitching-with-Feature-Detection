import cv2
import numpy as np
import os
import random
from support.feature_detection import feature_detector
from support.homography import dlt_homography, homography_coord_transformation, \
    L2_Norm_dist, RANSAC_homography
from support.undistortion import undistorting_homography_projections

# The image loader loads the correct set of images required to make the
# panorama.
def loading_images(directory):
    # The file extension for run 3.
    run_3 = '/run3_base_hr'
    specific_run = directory + run_3
    # The specific camera photos that we want to extract.
    target_list = ['omni_image1','omni_image3','omni_image5','omni_image7',
                   'omni_image9']
    # Stores the images used.
    img_list = []
    # Looking through the directories given to us in Run 3:
    for dir in os.listdir(specific_run):
        # If the directory is one of those we want:
        if dir in target_list:
            # Get the whole directory link of the last image (In this case, we
            # are using the 2nd last image since the last image isn't readable
            # in my case).
            last_img_filename = os.listdir(os.path.join(specific_run, dir))[-2]
            # Loading the image we want from the directory.
            img = cv2.imread(os.path.join(os.path.join(specific_run, dir),
                                          last_img_filename), cv2.IMREAD_COLOR)
            # Adding the image to the list of images.
            img_list.append(img)
    return img_list

# The images are stitched together into a panorama.
def stitch_panorama(my_images, H_undistort):
    # Stores the warped images for stitching.
    warped_imgs = []
    # Iterating through the images:
    for i, img in enumerate(my_images):
        # Getting the image shape in a tuple.
        img_shape = (img.shape[1], img.shape[0])
        # Warping the images with their respective homography matrix.
        warped_img = cv2.warpPerspective(img, H_undistort[i], img_shape)
        warped_img[:img.shape[0], :img.shape[1]] = img
        warped_imgs.append(warped_img)
    # Stitching together the warped images.
    stitcher = cv2.Stitcher_create()
    # Creating the panorama.
    _, panorama = stitcher.stitch(warped_imgs)
    # Returns the panorama.
    return panorama

# Puts all the components of the panorama-producing processes together.
def produce_panorama(loaded_imgs, output_dir):
    homography_matrix_list = []
    # Iterating through the selected images:
    for i in range(len(loaded_imgs) - 1):
        # Getting the matching features of two adjacent images.
        coord_pairs = feature_detector(loaded_imgs[i], loaded_imgs[i+1],'sift')
        # Calculating the homography matrix between the two images.
        H_matrix = RANSAC_homography(coord_pairs)
        # Adding the homography matrix to the list of matrices.
        homography_matrix_list.append(H_matrix)
    # Defining the middle image.
    mid_img = (len(loaded_imgs) // 2) - 1
    # Getting the optimal homography transform matrices needed to avoid
    # distortion when stitching the images.
    H_undistort = undistorting_homography_projections(homography_matrix_list,
                                                      mid_img)
    # Stitching the images together to form a panorama.
    panoImg = stitch_panorama(loaded_imgs, H_undistort)
    output_img_name = '/robot_panorama_run3_test.png'
    # Saving the panorama to the output folder.
    cv2.imwrite(output_dir + output_img_name, panoImg)


