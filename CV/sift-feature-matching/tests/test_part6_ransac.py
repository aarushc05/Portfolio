import pytest
import numpy as np
import logging
import cv2
from pathlib import Path
from vision.part6_ransac import (
    calculate_num_ransac_iterations,
    ransac_homography,
)
from vision.utils import load_image, get_matches, single2im

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def test_calculate_num_ransac_iterations():
    data_set = [
        (0.99, 1, 0.99, 1),
        (0.99, 10, 0.9, 11),
        (0.9, 15, 0.5, 75450),
        (0.95, 5, 0.66, 22),
    ]

    for prob_success, sample_size, ind_prob, num_samples in data_set:
        S = calculate_num_ransac_iterations(prob_success, sample_size, ind_prob)
        assert pytest.approx(num_samples, abs=1.0) == S


def test_ransac_homography():
    np.random.seed(0)
    img_a_path = f"{DATA_ROOT}/Notre_Dame/921919841_a30df938f2_o.jpg"
    img_b_path = f"{DATA_ROOT}/Notre_Dame/4191453057_c86028ce1f_o.jpg"

    pic_a = single2im(load_image(img_a_path))
    pic_b = single2im(load_image(img_b_path))
    n_feat = 5000

    points_2d_pic_a, points_2d_pic_b = get_matches(pic_a, pic_b, n_feat)
    H, _, _ = ransac_homography(points_2d_pic_a, points_2d_pic_b)

    expected_H = np.array(
        [
            [9.16672947e-01, 2.14390233e-02, 5.39697088e+01],
            [8.87046763e-02, 7.97638785e-01, 8.68876943e+01],
            [7.88827205e-05, -3.93789431e-05, 1.00000000e+00],
        ]
    )

    H /= H[2, 2]
    expected_H /= expected_H[2, 2]

    # This is the expected H if you do the simplest implementation
    expected_H_simple = np.array([[ 9.07288363e-01,  1.51742595e-02,  5.89910237e+01],
       [ 8.29783924e-02,  7.83434281e-01,  9.57924161e+01],
       [ 7.55603410e-05, -4.64513675e-05,  1.00000000e+00]])

    # This is the expected H if you do both distances
    expected_H_dual = np.array([[ 9.07288363e-01,  1.51742595e-02,  5.89910237e+01],
       [ 8.29783924e-02,  7.83434281e-01,  9.57924161e+01],
       [ 7.55603410e-05, -4.64513675e-05,  1.00000000e+00]])

    # This is the expected H if you do refinement but only A to B distance
    expected_H_refine = np.array([[ 9.18636676e-01,  2.17652288e-02,  5.37773426e+01],
       [ 9.03569719e-02,  8.00446783e-01,  8.45336969e+01],
       [ 8.00576365e-05, -3.84343462e-05,  1.00000000e+00]])

    assert np.allclose(H, expected_H, atol=10) or np.allclose(H, expected_H_dual, atol=5) or np.allclose(H, expected_H_refine, atol=5) or np.allclose(H, expected_H_simple, atol=5)
