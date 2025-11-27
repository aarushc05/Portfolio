import numpy as np


def assemble_matrix(points_2d: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Generates a matrix of the form:

    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1 -u1
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1 -v1
      .  .  .  . .  .  .  .    .     .      .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn -un
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn -vn ]

    given a set of 2D points (u,v) and 3D points (X,Y,Z).

    You will use this matrix to solve for the projection matrix with SVD,
    or with minor modification, NumPy's least squares implementation.

    Args:
        points_2d: A numpy array of shape (N, 2)
        points_3d: A numpy array of shape (N, 3)

    Returns:
        A: A matrix containing 2D and 3D points, as part of a system
    """
    A = []
    for i in range(points_2d.shape[0]):
        u, v = points_2d[i]
        X, Y, Z = points_3d[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v])
    return np.array(A)



def calculate_projection_matrix(
    points_2d: np.ndarray, points_3d: np.ndarray
) -> np.ndarray:
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                          [ M11      [ 0
                                                            M12        0
                                                            M13        .
                                                            M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1 -u1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1 -v1        M22        .
      .  .  .  . .  .  .  .    .     .      .           *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn -un        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn -vn ]      M31        .
                                                            M32        0
                                                            M33        0
                                                            M34]       0 ]

    Then you can solve this using SVD or least squares with np.linalg.lstsq().
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs. If you are using
    least squares, you will need to modify the last column of the matrix to
    avoid a degenerate solution (Hint: One method uses a homogenous linear
    system while the other uses nonhomogenous).

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    A = assemble_matrix(points_2d, points_3d)
    U, S, Vt = np.linalg.svd(A)
    m = Vt[-1, :]
    M = m.reshape(3, 4)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return M


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Computes projection from [X,Y,Z] in non-homogenous coordinates to
    (x,y) in non-homogenous image coordinates. Performed via multiplying
    the projection matrix and 3D points vectors. Remember to divide each
    projected point by the homogenous coordinate to get the final set of
    image coordinates.

    Args:
        P: 3 x 4 projection matrix
        points_3d: n x 3 array of points [X_i,Y_i,Z_i]
    Returns:
        projected_points_2d: n x 2 array of points in non-homogenous image
            coordinates
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    n = points_3d.shape[0]
    pts_h = np.hstack([points_3d, np.ones((n, 1))])
    proj_h = (P @ pts_h.T).T
    projected_points_2d = proj_h[:, :2] / proj_h[:, 2:3]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return projected_points_2d


def calculate_camera_center(M: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix. Equations 5
    and 6 from the documentation will be helpful here.

    A useful method will be np.linalg.inv().

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    Q = M[:, :3]
    q = M[:, 3]
    cc = -np.linalg.inv(Q) @ q
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return cc
