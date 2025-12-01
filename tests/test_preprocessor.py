import cv2
import numpy as np
import pytest
from src.features.processing import ImagePreprocessor

@pytest.fixture
def preprocessor():
    return ImagePreprocessor(target_width=200, target_height=141)

def test_check_blur_good_image(preprocessor):
    # Sharp image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 80), (255, 255, 255), -1)
    # Laplacian variance should be high, so check_blur returns False (not blurry)
    # Need to ensure variance > threshold (100.0)
    # A sharp rectangle on black background has very high variance.
    assert preprocessor.check_blur(image) is False

def test_check_blur_bad_image(preprocessor):
    # Flat image (variance 0)
    image = np.zeros((100, 100, 3), dtype=np.uint8) + 128
    # Variance is 0 < 100, so check_blur returns True (blurry)
    assert preprocessor.check_blur(image) is True

def test_find_paper_contour(preprocessor):
    # Create an image with a clear white rectangle on black background
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    # Rotated rectangle somewhat
    pts = np.array([[50, 50], [250, 50], [250, 200], [50, 200]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts], (255, 255, 255))

    contour = preprocessor.find_paper_contour(image)
    assert contour.shape == (4, 2)

def test_slice_grid_structure(preprocessor):
    # Mock warped image (200x141 as per fixture)
    warped = np.zeros((141, 200), dtype=np.uint8)

    squares = preprocessor.slice_grid(warped)

    # 8 squares
    assert len(squares) == 8
    # Check keys
    for i in range(1, 9):
        assert i in squares

    # Check dimensions
    # cell_w = 200 // 4 = 50
    # cell_h = 141 // 3 = 47
    # margin_x = 5, margin_y = 4
    # w = 50 - 10 = 40
    # h = 47 - 8 = 39

    s1 = squares[1]
    assert s1.shape == (39, 40)
