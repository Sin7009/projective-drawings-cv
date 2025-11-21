import pytest
import numpy as np
import cv2
from src.features.extraction import FeatureExtractor

class TestFeatureExtractor:

    def test_extract_square_1_features(self):
        # Create 100x100 black image
        image = np.zeros((100, 100), dtype=np.uint8)

        # Draw a dot at (75, 25) -> Top-Right quadrant
        # In image coords: x=75, y=25.
        # Center is (50, 50).
        # dx = 75 - 50 = 25
        # dy = 25 - 50 = -25
        image[25, 75] = 255

        features = FeatureExtractor.extract_square_1_features(image)

        assert features['centroid_x'] == 75.0
        assert features['centroid_y'] == 25.0

        # Distance = sqrt(25^2 + (-25)^2) = sqrt(1250) approx 35.35
        expected_dist = np.sqrt(25**2 + 25**2)
        assert abs(features['displacement_distance'] - expected_dist) < 0.1

        # Angle: atan2(-25, 25) = -45 degrees
        assert abs(features['displacement_angle'] - (-45.0)) < 0.1

    def test_extract_square_3_features_uphill(self):
        # Square 3: Ambition. High positive slope.
        # Create image with "uphill" line.
        # Visually uphill (/) means x increases, y decreases (in image coords).
        # (0, 100) to (100, 0).
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(image, (0, 99), (99, 0), 255, 1)

        features = FeatureExtractor.extract_square_3_features(image)

        # Slope in image coords: (0 - 99) / (99 - 0) = -1
        # Verify slope is close to -1
        assert abs(features['slope'] - (-1.0)) < 0.1

    def test_extract_square_4_features(self):
        # Square 4: Density
        # 10x10 image, 50 pixels white
        image = np.zeros((10, 10), dtype=np.uint8)
        image[0:5, :] = 255 # Top half white

        features = FeatureExtractor.extract_square_4_features(image)

        assert features['pixel_density'] == 0.5

        # Test ROI (Top Right)
        # Image is 10x10. ROI x > 7, y < 3.
        # Our drawing is top half (y 0-4).
        # ROI is in drawn area.
        # Check heavily_shaded logic.
        # ROI density should be 1.0 (since top half is all white).
        assert features['roi_density'] == 1.0
        assert features['heavily_shaded'] is True

    def test_extract_square_5_features_intersection(self):
        # Square 5: Intersections.
        # Draw a cross (+).
        image = np.zeros((100, 100), dtype=np.uint8)

        # Horizontal line
        cv2.line(image, (20, 50), (80, 50), 255, 2)
        # Vertical line
        cv2.line(image, (50, 20), (50, 80), 255, 2)

        features = FeatureExtractor.extract_square_5_features(image)

        # Should detect at least 2 lines (hough might detect fragments, but at least 2 main ones)
        assert features['line_count'] >= 2
        # They should intersect
        assert features['intersection_count'] >= 1
