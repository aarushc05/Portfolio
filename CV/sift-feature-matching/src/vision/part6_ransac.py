import math

import numpy as np
import cv2


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: int
) -> int:
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float representing the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    w = ind_prob_correct
    s = sample_size
    if w == 1.0:
        return 1
    denom = 1 - (w ** s)
    if denom <= 0:
        return 1
    num = 1 - prob_success
    if num <= 0:
        return 1
    num_samples = np.log(num) / np.log(1 - (w ** s))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(np.ceil(num_samples))


def ransac_homography(points_a: np.ndarray, points_b: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Uses the RANSAC algorithm to robustly estimate a homography matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) of points from image A.
    -   points_b: A numpy array of shape (N, 2) of corresponding points from image B.

    Returns:
    -   best_H: The best homography matrix of shape (3, 3).
    -   inliers_a: The subset of points_a that are inliers (M, 2).
    -   inliers_b: The subset of points_b that are inliers (M, 2).
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    #                                                                         #
    # HINT: You are allowed to use the `cv2.findHomography` function to       #
    # compute the homography from a sample of points. To compute a direct     #
    # solution without OpenCV's built-in RANSAC, use it like this:            #
    #   H, _ = cv2.findHomography(sample_a, sample_b, 0)                      #
    # The `0` flag ensures it computes a direct least-squares solution.       #
    ###########################################################################

    prob_success = 0.999
    sample_size = 4
    outlier_ratio = 0.5
    inlier_pixel_threshold = 5.0
    # calculate the number of samples needed
    numSamples = calculate_num_ransac_iterations(prob_success, sample_size, outlier_ratio)

    best_H = None
    max_inliers = 0
    inliers_a = np.array([])
    inliers_b = np.array([])

    # reshape once for perspective transform
    reshaped_points_a = points_a.reshape(-1, 1, 2)

    for _ in range(numSamples):
        # sample points randomly
        index = np.random.choice(points_a.shape[0], sample_size, replace=False)
        sampleA = points_a[index]
        sampleB = points_b[index]
        
        # compute homography from samples
        H, _ = cv2.findHomography(sampleA, sampleB, 0)

        # score model
        projected_points = cv2.perspectiveTransform(reshaped_points_a, H).reshape(-1, 2)
        errors = np.linalg.norm(projected_points - points_b, axis=1)
        inliers = np.where(errors < inlier_pixel_threshold)[0]
        
        # update best model if needed
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_H = H

    if max_inliers > 0:
        # recompute inliers for best homography
        projected_points = cv2.perspectiveTransform(reshaped_points_a, best_H).reshape(-1, 2)
        errors = np.linalg.norm(projected_points - points_b, axis=1)
        inliers = np.where(errors < inlier_pixel_threshold)[0]
        inliers_a = points_a[inliers]
        inliers_b = points_b[inliers]

        # update best homography if needed
        if len(inliers) >= sample_size:
            best_H, _ = cv2.findHomography(inliers_a, inliers_b, 0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_H, inliers_a, inliers_b