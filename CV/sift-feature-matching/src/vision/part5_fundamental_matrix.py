"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    pts = points.astype(np.float32)
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    std = np.std(centered, axis=0)
    std[std == 0] = 1
    S = np.array([[1.0 / std[0], 0, 0], [0, 1.0 / std[1], 0], [0, 0, 1]])
    T_trans = np.array([[1, 0, -mean[0]], [0, 1, -mean[1]], [0, 0, 1]])
    T = S @ T_trans
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    ptsn_h = (T @ pts_h.T).T
    points_normalized = ptsn_h[:, :2]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    F_orig = T_b.T @ F_norm @ T_a
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def make_singular(F_norm: np.array) -> np.ndarray:
    """
    Force F to be singular by zeroing the smallest of its singular values.
    This is done because F is not supposed to be full rank, but an inaccurate
    solution may end up as rank 3.

    Args:
    - F_norm: A numpy array of shape (3,3) representing the normalized fundamental matrix.

    Returns:
    - F_norm_s: A numpy array of shape (3, 3) representing the normalized fundamental matrix
                with only rank 2.
    """
    U, D, Vt = np.linalg.svd(F_norm)
    D[-1] = 0
    F_norm_s = np.dot(np.dot(U, np.diag(D)), Vt)

    return F_norm_s


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here. Equation (9) in the documentation indicates
    one equation of a linear system in which you'll want to solve for f_{i, j}.

    Since the matrix is defined up to a scale, many solutions exist. To constrain
    your solution, use can either use SVD and use the last Vt vector as your
    solution, or you can fix f_{3, 3} to be 1 and solve with least squares.

    Be sure to reduce the rank of your estimate - it should be rank 2. The
    make_singular() function can do this for you.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    pa_n, Ta = normalize_points(points_a)
    pb_n, Tb = normalize_points(points_b)
    N = pa_n.shape[0]
    A = []
    for i in range(N):
        xa, ya = pa_n[i]
        xb, yb = pb_n[i]
        A.append([xb * xa, xb * ya, xb, yb * xa, yb * ya, yb, xa, ya, 1])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    f = Vt[-1, :]
    Fn = f.reshape(3, 3)
    Fn = make_singular(Fn)
    F = unnormalize_F(Fn, Ta, Tb)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
