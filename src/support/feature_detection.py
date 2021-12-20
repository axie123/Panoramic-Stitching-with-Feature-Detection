import cv2

# This function conducts identical feature pair detection between images.
def feature_detector(left_img, right_img, detector):
    if detector == 'sift':
        # Loads the SIFT feature detection algorithm with 3 Gaussian kernels
        # for the Gaussian pyramid.
        sift_detector = cv2.SIFT_create(0, 3, 0)
        # Computing the features and descriptors on the left img.
        l_keypoints, l_descriptors = sift_detector.detectAndCompute(left_img,
                                                                    None)
        # Computing the features and descriptors on the right img.
        r_keypoints, r_descriptors = sift_detector.detectAndCompute(right_img,
                                                                    None)
    elif detector == 'surf':
        # Loads the SURF feature detection algorithm with 3 Gaussian kernels
        # for the Gaussian pyramid.
        surf_detector = cv2.SURF_create(0, 3, 0)
        # Computing the features and descriptors on the left img.
        l_keypoints, l_descriptors = surf_detector.detectAndCompute(left_img,
                                                                    None)
        # Computing the features and descriptors on the right img.
        r_keypoints, r_descriptors = surf_detector.detectAndCompute(right_img,
                                                                    None)
    # Using a brute force descriptor matcher using L2 norm metric.
    descriptor_matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    # Identifying identical features between the two images.
    mutual_matches = descriptor_matcher.match(l_descriptors, r_descriptors)

    matching_features = []
    # Iterating through the identical features.
    for m in mutual_matches:
        # Limiting the distance between the features to 70.
        if m.distance <= 70:
            left_img_i, right_img_i = m.queryIdx, m.trainIdx
            # Extracting the key points of the features from both images.
            x_l, y_l = l_keypoints[left_img_i].pt
            x_r, y_r = r_keypoints[right_img_i].pt
            # Adding the qualifying matching feature coordinates to the list.
            matching_features.append([x_l, y_l, x_r, y_r])
    return matching_features