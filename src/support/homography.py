import numpy as np
import random

def dlt_homography(I1pts, I2pts):
    """
    Finds the perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    -----------
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    # We first want to initialize the point correspondence matrix for the 4
    # correspondence points.
    A = np.zeros((8, 9))

    # The correspondence points are loaded.
    correspondence = I2pts

    # Each point from Image 1 has its 1-to-1 correspondence point. This can be
    # described by the equation c(u v 1) = H*(x y 1), with (u, v, 1) is the
    # correspondence to (x, y, 1). We can put these points in the correspondence
    # matrix A.
    for i, coord in enumerate(I1pts.T):
        # Load the correspondence point for the coordinate.
        specific_corres = correspondence[:,i]
        u, v = specific_corres[0], specific_corres[1]
        # Load the pixel coordinate from Image 1.
        x = coord[0]
        y = coord[1]

        # Adding the 2 x 9 matrix for each correspondence point in the overall
        # correspondence matrix. We can calculate the H matrix by taking the
        # null space of A.
        A_i = np.array([[-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                        [0, 0, 0, -x, -y, -1, v*x, v*y, v]])

        # Placing the 2 x 9 matrix for the point into its appropriate position
        # within the correspondence matrix.
        A[i*2:2*i + 2,:] = A_i

    _, _, M = np.linalg.svd(A)

    # Puts the minimum singular array into a 3x3 matrix as the homography.
    H = M[8].reshape(3, 3)

    # Normalizes the homography matrix.
    norm_entry = H[2][2]
    H *= 1 / norm_entry
    return H

# Calculates the corresponding coordinates of the second/right image with
# the homography matrix.
def homography_coord_transformation(coordinates, H):
    homography_image_corres = []
    # Iterating through the coordinates of the first image:
    for i, c in enumerate(coordinates):
        # Defining the feature coordinates of the first image.
        u, v = c[0], c[1]
        # Convert to augmented points, apply homography to get
        # correspondences to the second image.
        correspondence = np.dot(H, np.array([u, v, 1.0]).T)
        # Normalizing the correspondence matrix.
        correspondence /= correspondence[2]
        # Adding the correspondence coordinates to the list of image
        # correspondences calculated by the homography.
        corres_coords = [correspondence[0], correspondence[1]]
        homography_image_corres.append(corres_coords)
    return homography_image_corres

# Using the L2 Norm metric to determine which features coords are inliers.
def L2_Norm_dist(coordinates, H):
    # Stores all the inlier feature coordinates.
    inlier_coords = []
    # Calculates the corresponding coordinates of the second/right image with
    # the homography matrix.
    homography_correspondences = homography_coord_transformation(coordinates, H)
    # Iterating through the feature coordinates:
    for i, coord in enumerate(coordinates):
        # Loading the coordinates of the features on the right/second image.
        x, y = coord[2], coord[3]
        # Loading the coordinates of the features on the right/second image
        # calculated with the homography matrix.
        u, v = homography_correspondences[i][0], homography_correspondences[i][1]

        # Calculates the L2 Norm distance between the homography correspondence
        # point and the actual location of the feature on th second image.
        error = np.linalg.norm(np.array([x, y, 1.0]).T - np.array([u, v, 1.0]).T)
        # We set the L2 Norm distance tolerance to be 5 in this case.
        if error < 5:
            inlier_coords.append(coordinates[i])
    return inlier_coords

# Implements the RANSAC algorithm to calculate the homography matrices for
# feature projection.
def RANSAC_homography(coordinates):
    # Stores the inlier feature points and optimal homography matrix.
    max_inlier_features = []
    opt_homography_matrix = None

    for i in range(1200):
        # Randomly selecting 4 pairs of matching feature coordinates.
        rdm_idx_1 = random.randrange(0, len(coordinates))
        rdm_idx_2 = random.randrange(0, len(coordinates))
        rdm_idx_3 = random.randrange(0, len(coordinates))
        rdm_idx_4 = random.randrange(0, len(coordinates))

        # Selects the 4 pairs of the points.
        rdm_pt_1, rdm_pt_2 = coordinates[rdm_idx_1], coordinates[rdm_idx_2]
        rdm_pt_3, rdm_pt_4 = coordinates[rdm_idx_3], coordinates[rdm_idx_4]
        Ipts = np.array([rdm_pt_1, rdm_pt_2, rdm_pt_3, rdm_pt_4])

        # Dividing up the feature coordinates between the left and right imgs.
        I1pts, I2pts = Ipts[:,:2].T, Ipts[:,2:].T

        # Calculates the homography matrix between the two images.
        H = dlt_homography(I1pts,I2pts)

        # Get all the inlier coordinates that satisfy the L2 norm.
        inlier_pts = L2_Norm_dist(coordinates, H)

        # If there are more points that fit within the L2 Norm tolerance
        # than the current maximum, update the homography matrix and list of
        # inlier features.
        if len(inlier_pts) > len(max_inlier_features):
            max_inlier_features = inlier_pts
            opt_homography_matrix = H

        # If 75% of all features fit within the tolerance, stop.
        if len(max_inlier_features) > (len(coordinates) * 0.75):
            break

    return opt_homography_matrix
