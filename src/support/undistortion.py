import numpy as np

# This calculates the homography projection of the features from the component
# images relative to those of the central image of the panorama. This is done
# to make sure that there is no image distortion towards the ends of the
# resultant panorama (1st and last image).
def undistorting_homography_projections(homography_matrix_list, mid):
    # Stores all the homography projection matrices.
    H_undistort = []
    i = 0

    # Checking if the index is less than the length of the homography matrix.
    while i <= len(homography_matrix_list):
        # If the index is smaller than the midpoint:
        if i < mid:
            # Extract the homography matrix that directly maps to the central
            # component image from the image left of it.
            h_proj = homography_matrix_list[mid]
            j = mid - 1
            # Calculates the homography matrices used to project and warp
            # features from all images previous to the central image to the
            # central image that are not directly connected.
            while i <= j:
                h_proj = np.matmul(h_proj, homography_matrix_list[j])
                j -= 1
            # Adding the projecting matrix to the list of final projecting
            # matrices.
            H_undistort.append(h_proj)
            i += 1
        # If the index is equal to the midpoint:
        elif i == mid:
            # In this case, we are want to project and warp the features from
            # the image left of the central image to the central image. Since
            # the projection is already available, we use the identity matrix.
            h_proj = np.identity(3)
            # Adding the projecting matrix to the list of final projecting
            # matrices.
            H_undistort.append(h_proj)
            i += 1
        # If the image index is to the right of the midpoint:
        else:
            j = mid + 1
            # Since we are projecting right to left, we need the inverse
            # matrix.
            h_proj = np.linalg.inv(homography_matrix_list[mid])
            # Calculates the homography matrices used to project and warp
            # features from all images after to the central image to the
            # central image that are not directly connected. We would need to
            # take the inverse of the homography matrices since we are
            # projecting backwards.
            while i > j:
                invH = np.linalg.inv(homography_matrix_list[j])
                h_proj = np.matmul(h_proj, invH)
                j += 1
            # Adding the projecting matrix to the list of final projecting
            # matrices.
            H_undistort.append(h_proj)
            i += 1
    return H_undistort