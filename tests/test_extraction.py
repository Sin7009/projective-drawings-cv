
import pytest
import numpy as np
import cv2
from src.features.extraction import FeatureExtractor

class TestFeatureExtractor:

    def create_dummy_image(self, type_name):
        img = np.zeros((100, 100), dtype=np.uint8)

        if type_name == "circle":
            cv2.circle(img, (50, 50), 30, 255, 2) # Closed, smooth
        elif type_name == "star":
            # Draw a star-like shape (wavy/jagged)
            pts = np.array([[50, 10], [60, 40], [90, 40], [65, 60], [75, 90], [50, 70], [25, 90], [35, 60], [10, 40], [40, 40]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, 255, 2)
        elif type_name == "open_lines":
            cv2.line(img, (10, 10), (90, 90), 255, 2)
            cv2.line(img, (90, 10), (10, 90), 255, 2)
        elif type_name == "dots":
            # 8 dots
            for i in range(8):
                cv2.circle(img, (10 + i*10, 50), 2, 255, -1)
        elif type_name == "connected_dots":
            # Line connecting where dots would be
            cv2.line(img, (10, 50), (90, 50), 255, 2)
        elif type_name == "under_arc":
            # Arc at top (though not drawn, we assume user drawing is below)
            # Drawing below
            cv2.rectangle(img, (30, 60), (70, 90), 255, -1)
        elif type_name == "over_arc":
            # Drawing above (small y)
            cv2.line(img, (50, 10), (50, 40), 255, 2)

        return img

    def test_extract_square_2_features(self):
        # Smoothness
        img_circle = self.create_dummy_image("circle")
        res_circle = FeatureExtractor.extract_square_2_features(img_circle)
        assert res_circle['smoothness'] > 0.9 # Circle is very smooth

        img_star = self.create_dummy_image("star")
        res_star = FeatureExtractor.extract_square_2_features(img_star)
        assert res_star['smoothness'] < res_circle['smoothness']

    def test_extract_square_6_features(self):
        # Closed shape
        img_closed = self.create_dummy_image("circle")
        res_closed = FeatureExtractor.extract_square_6_features(img_closed)
        assert res_closed['is_closed_shape'] is True

        img_open = self.create_dummy_image("open_lines")
        res_open = FeatureExtractor.extract_square_6_features(img_open)
        assert res_open['is_closed_shape'] is False

    def test_extract_square_7_features(self):
        # Connected dots
        img_dots = self.create_dummy_image("dots")
        res_dots = FeatureExtractor.extract_square_7_features(img_dots)
        assert res_dots['dots_connected'] is False
        assert res_dots['contour_count'] >= 5

        img_conn = self.create_dummy_image("connected_dots")
        res_conn = FeatureExtractor.extract_square_7_features(img_conn)
        assert res_conn['dots_connected'] is True
        assert res_conn['contour_count'] <= 4

    def test_extract_square_8_features(self):
        # Under Arc
        img_under = self.create_dummy_image("under_arc")
        res_under = FeatureExtractor.extract_square_8_features(img_under)
        assert res_under['is_under_arc'] is True
        assert res_under['centroid_y_norm'] > 0.4

        img_over = self.create_dummy_image("over_arc")
        res_over = FeatureExtractor.extract_square_8_features(img_over)
        assert res_over['is_under_arc'] is False
        assert res_over['centroid_y_norm'] < 0.4
