""" CS4277/CS5477 Lab 1: Metric Rectification and Robust Homography Estimation.
See accompanying file (lab1.pdf) for instructions.

Name: Wong Yuk Kwan
Email: e0771093@u.nus.edu
Student ID: A0237497M
"""

import numpy as np
import cv2
from helper import *
from math import floor, ceil, sqrt, cos, sin



def compute_homography(src, dst):
    """Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    """

    h_matrix = np.eye(3, dtype=np.float64)

    """ YOUR CODE STARTS HERE """
    # Compute normalization matrix
    centroid_src = np.mean(src, axis=0)
    d_src = np.linalg.norm(src - centroid_src[None, :], axis=1)
    s_src = sqrt(2) / np.mean(d_src)
    T_norm_src = np.array([[s_src, 0.0, -s_src * centroid_src[0]],
                           [0.0, s_src, -s_src * centroid_src[1]],
                           [0.0, 0.0, 1.0]])

    centroid_dst = np.mean(dst, axis=0)
    d_dst = np.linalg.norm(dst - centroid_dst[None, :], axis=1)
    s_dst = sqrt(2) / np.mean(d_dst)
    T_norm_dst = np.array([[s_dst, 0.0, -s_dst * centroid_dst[0]],
                           [0.0, s_dst, -s_dst * centroid_dst[1]],
                           [0.0, 0.0, 1.0]])

    srcn = transform_homography(src, T_norm_src)
    dstn = transform_homography(dst, T_norm_dst)

    # Compute homography
    n_corr = srcn.shape[0]
    A = np.zeros((n_corr*2, 9), dtype=np.float64)
    for i in range(n_corr):
        A[2 * i, 0] = srcn[i, 0]
        A[2 * i, 1] = srcn[i, 1]
        A[2 * i, 2] = 1.0
        A[2 * i, 6] = -dstn[i, 0] * srcn[i, 0]
        A[2 * i, 7] = -dstn[i, 0] * srcn[i, 1]
        A[2 * i, 8] = -dstn[i, 0] * 1.0

        A[2 * i + 1, 3] = srcn[i, 0]
        A[2 * i + 1, 4] = srcn[i, 1]
        A[2 * i + 1, 5] = 1.0
        A[2 * i + 1, 6] = -dstn[i, 1] * srcn[i, 0]
        A[2 * i + 1, 7] = -dstn[i, 1] * srcn[i, 1]
        A[2 * i + 1, 8] = -dstn[i, 1] * 1.0

    u, s, vt = np.linalg.svd(A)
    h_matrix_n = np.reshape(vt[-1, :], (3, 3))

    # Unnormalize homography
    h_matrix = np.linalg.inv(T_norm_dst) @ h_matrix_n @ T_norm_src
    h_matrix /= h_matrix[2, 2]

    # src = src.astype(np.float32)
    # dst = dst.astype(np.float32)
    # h_matrix = cv2.findHomography(src, dst)[0].astype(np.float64)
    """ YOUR CODE ENDS HERE """

    return h_matrix


def transform_homography(src, h_matrix):
    """Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    Prohibited functions:
        cv2.perspectiveTransform()

    """
    transformed = None

    """ YOUR CODE STARTS HERE """
    
    # Homogenize src points
    n, _ = src.shape
    homo_column = np.full((n, 1), 1)
    src = np.column_stack((src, homo_column))
    
    # Apply homography matrix
    src = np.transpose(src)
    transformed = np.matmul(h_matrix, src)
    
    # Dehomogenize transformed point
    de_homo = transformed[-1]
    transformed = transformed / de_homo
    transformed = transformed[:-1]
    transformed = np.transpose(transformed)
    
    """ YOUR CODE ENDS HERE """

    return transformed


def warp_image(src, dst, h_matrix):
    """Applies perspective transformation to source image to warp it onto the
    destination (background) image

    Args:
        src (np.ndarray): Source image to be warped
        dst (np.ndarray): Background image to warp template onto
        h_matrix (np.ndarray): Warps coordinates from src to the dst, i.e.
                                 x_{dst} = h_matrix * x_{src},
                               where x_{src}, x_{dst} are the homogeneous
                               coordinates in I_{src} and I_{dst} respectively

    Returns:
        dst (np.ndarray): Source image warped onto destination image

    Prohibited functions:
        cv2.warpPerspective()
    You may use the following functions: np.meshgrid(), cv2.remap(), transform_homography()
    """
    dst = dst.copy()  # deep copy to avoid overwriting the original image

    """ YOUR CODE STARTS HERE """

    # Construct image coordinate grid
    m, n, _ = dst.shape
    x_axis = np.arange(n, dtype=np.float32)
    y_axis = np.arange(m, dtype=np.float32)
    map_x, map_y = np.meshgrid(x_axis, y_axis)

    # Get the list of coordinates
    points = np.dstack((map_x, map_y))
    points = points.reshape((m*n, 2))

    # Transform the point by inverse of h_matrix
    h_inv = np.linalg.inv(h_matrix)
    transformed = transform_homography(points, h_inv)

    # Wrap template image into correct location
    map_x = transformed[:, 0].reshape((m, n)).astype(np.float32)
    map_y = transformed[:, 1].reshape((m, n)).astype(np.float32)
    wrapped_src = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_TRANSPARENT)
    
    # Cover the dst image by the wrapped template
    black_pixel = np.array([0, 0, 0])
    for i in range(m):
        for j in range(n):
            if not np.array_equal(wrapped_src[i, j, :], black_pixel):
                dst[i, j, :] = wrapped_src[i, j, :]

    """ YOUR CODE ENDS HERE """
    # cv2.warpPerspective(src, h_matrix, dsize=dst.shape[1::-1],
    #                     dst=dst, borderMode=cv2.BORDER_TRANSPARENT)
    return dst

def compute_affine_rectification(src_img:np.ndarray,lines_vec: list):
    '''
       The first step of the stratification method for metric rectification. Compute
       the projective transformation matrix Hp with line at infinity. At least two
       parallel line pairs are required to obtain the vanishing line. Then warping
       the image with the predicted projective transformation Hp to recover the affine
       properties. X_dst=Hp*X_src

       Args:
           src_img: Original image X_src
           lines_vec: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
       Returns:
           Xa: Affinely rectified image by removing projective distortion

    '''
    dst = np.zeros_like(src_img)  # deep copy to avoid overwriting the original image
    Hp= np.zeros((3,3))
    """ YOUR CODE STARTS HERE """
    # Note that lines_vec[0] and lines_vec[1] are parallel to each other
    # Note that lines_vec[2] and lines_vec[3] are parallel to each other
    
    # Compute line at infinity
    line0, line1 = lines_vec[0], lines_vec[1]
    intersect_pt1 = line0.intersetion_point(line1)

    line2, line3 = lines_vec[2], lines_vec[3]
    intersect_pt2 = line2.intersetion_point(line3)

    line_inf = Line(intersect_pt1, intersect_pt2)
    l1, l2, l3 = line_inf.vec_para

    # Compute the Hp_prime
    Hp_T =  np.matrix([[1, 0, -l1/l3],
                       [0, 1, -l2/l3],
                       [0, 0, 1/l3]])

    Hp = np.transpose(Hp_T)
    Hp_inv = np.linalg.inv(Hp)
    Hp_prime = Hp_inv   # Assume that affinity matrix is identity matrix

    # Scaling the result so that it is not out of bound
    scale = 0.2
    scaling_matrix = np.matrix([[scale, 0, 0],
                                [0, scale, 0],
                                [0, 0, 1]])

    Hp_prime = np.matmul(scaling_matrix, Hp_prime)
    
    # print(Hp_prime)

    # Finally warp the image
    dst = warp_image(src_img,dst, Hp_prime)
    """ YOUR CODE ENDS HERE """
   
    return dst



def compute_metric_rectification_step2(src_img:np.ndarray,line_vecs: list):
    '''
       The second step of the stratification method for metric rectification. Compute
       the affine transformation Ha with the degenerate conic from at least two
       orthogonal line pairs. Then warping the image with the predicted affine
       transformation Ha to recover the metric properties. X_dst=Ha*X_src

       Args:
           src_img: Affinely rectified image X_src
           line_vecs: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
       Returns:
           X_dst: Image after metric rectification

    '''
    dst = np.zeros_like(src_img)  # deep copy to avoid overwriting the original image
    Ha = np.zeros((3, 3))
    """ YOUR CODE STARTS HERE """

    # Note that line0 is orthogonal to line1 and
    # line2 is orthogonal to line3
    line0, line1 = line_vecs[0], line_vecs[1]
    line2, line3 = line_vecs[2], line_vecs[3]

    l11, l12, _ = line0.vec_para
    m11, m12, _ = line1.vec_para

    l21, l22, _ = line2.vec_para
    m21, m22, _ = line3.vec_para

    A = np.matrix([[l11*m11, l11*m12 + l12*m11, l12*m12],
                   [l21*m21, l21*m22 + l22*m21, l22*m22]])

    _, _ , V = np.linalg.svd(A)
    s = V[-1, :]
    
    S = np.matrix([[s.item((0, 0)), s.item((0, 1))],
                   [s.item((0, 1)), s.item((0, 2))]])

    K = np.linalg.cholesky(S)

    Ha = np.matrix([[K.item((0, 0)), K.item((0, 1)), 0],
                    [K.item((1, 0)), K.item((1, 1)), 0],
                    [0, 0, 1]])

    Ha = np.linalg.inv(Ha)

    scaling = 0.5
    offset = [50, 150]
    scaling_matrix = np.matrix([[scaling, 0, offset[0]],
                                [0, scaling, offset[1]],
                                [0, 0, 1]])

    Ha = np.matmul(scaling_matrix, Ha)

    dst = warp_image(src_img, dst, Ha)
    """ YOUR CODE ENDS HERE """

  
    return dst

def compute_metric_rectification_one_step(src_img:np.ndarray,line_vecs: list):
    '''
       One-step metric rectification. Compute the transformation matrix H (i.e. H=HaHp) directly
       from five orthogonal line pairs. Then warping the image with the predicted affine
       transformation Ha to recover the metric properties. X_dst=H*X_src
       Args:
           src_img: Original image Xc
           line_infinity: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
       Returns:
           Xa: Image after metric rectification

    '''
    dst = np.zeros_like(src_img)  # deep copy to avoid overwriting the original image
    H = np.zeros((3, 3))

    """ YOUR CODE STARTS HERE """

    M = np.zeros((5, 6))
    idx = 0
    for i in range(0, len(line_vecs), 2):
        line0 = line_vecs[i]
        line1 = line_vecs[i+1]

        l1, l2, l3 = line0.vec_para
        m1, m2, m3 = line1.vec_para

        temp = np.array([l1*m1, (l1*m2+l2*m1)/2, l2*m2, (l1*m3 + l3*m1)/2, (l2*m3 + l3*m2)/2, l3*m3])
        M[idx, :] = temp
        idx += 1

    _, _, V = np.linalg.svd(M)
    c_vec = V[-1, :]

    a, b, c, d, e, f = c_vec

    C = np.matrix([[a, b/2, d/2],
                   [b/2, c, e/2],
                   [d/2, e/2, f]])

    U, S, _ = np.linalg.svd(C)
    
    A = np.matrix([[1/sqrt(S[0]), 0, 0],
                   [0, 1/sqrt(S[1]), 0],
                   [0, 0, 1]])
    
    H = np.linalg.inv(U)
    H = np.matmul(A, H)
    
    offset = [500, 50]
    scale = 0.1
    offset_matrix = np.matrix([[scale, 0, offset[0]],
                               [0, scale, offset[1]],
                               [0, 0, 1]])

    H = np.matmul(offset_matrix, H)

    dst = warp_image(src_img, dst, H)
    """ YOUR CODE ENDS HERE """

    return dst


def compute_homography_error(src, dst, homography):
    """Compute the squared bidirectional pixel reprojection error for
    provided correspondences

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        homography (np.ndarray): Homography matrix that transforms src to dst.

    Returns:
        err (np.ndarray): Array of size (N, ) containing the error d for each
        correspondence, computed as:
          d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
        where ||a|| denotes the l2 norm (euclidean distance) of vector a.
    """
    d = np.zeros(src.shape[0], np.float64)

    """ YOUR CODE STARTS HERE """
    
    # Calculate the forward error
    homography_inv = np.linalg.inv(homography)
    transformed_x_prime = transform_homography(dst, homography_inv)
    diff_forward = src - transformed_x_prime
    d_forward = np.linalg.norm(diff_forward, axis=1)
    d_forward = np.power(d_forward, 2)
    
    # Calculate the backword error
    transformed_x = transform_homography(src, homography)
    diff_backward = dst - transformed_x
    d_backward = np.linalg.norm(diff_backward, axis=1)
    d_backward = np.power(d_backward, 2)
    
    d = d_forward + d_backward

    """ YOUR CODE ENDS HERE """

    return d


def compute_homography_ransac(src, dst, thresh=16.0, num_tries=200):
    """Calculates the perspective transform from at least 4 points of
    corresponding points in a robust manner using RANSAC. After RANSAC, all the
    inlier correspondences will be used to re-estimate the homography matrix.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        thresh (float): Maximum allowed squared bidirectional pixel reprojection
          error to treat a point pair as an inlier (default: 16.0). Pixel
          reprojection error is computed as:
            d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
          where ||a|| denotes the l2 norm (euclidean distance) of vector a.
        num_tries (int): Number of trials for RANSAC

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.
        mask (np.ndarraay): Output mask with dtype np.bool where 1 indicates
          inliers

    Prohibited functions:
        cv2.findHomography()
    """

    h_matrix = np.eye(3, dtype=np.float64)
    mask = np.ones(src.shape[0], dtype=np.bool)

    """ YOUR CODE STARTS HERE """

    # Number of points
    s = 4       

    x_count         = src.shape[0] # Number of points in src
    x_prime_count   = dst.shape[0] # Number of points in dst

    max_inlier_count = 0           # Maximum number of inlier

    # RANSAC
    for _ in range(num_tries):
        
        # Select randome sample of correspondence
        rand_idx        = np.random.choice(x_count, size=s, replace=False)
        x_list          = src[rand_idx, :]
        x_prime_list    = dst[rand_idx, :]
        
        # Estimate error
        H = compute_homography(x_list, x_prime_list)
        error = compute_homography_error(src, dst, H)

        # Filter outliers
        M = np.ma.masked_less_equal(error, thresh)
        count = np.ma.count_masked(M)
        if count > max_inlier_count:
            mask = M.mask
            max_inlier_count = count

    # Final list without outlier
    final_x_list = src[mask, :]
    final_x_prime_list = dst[mask, :]

    h_matrix = compute_homography(final_x_list, final_x_prime_list)

    """ YOUR CODE ENDS HERE """

    return h_matrix, mask


