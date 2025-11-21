import unittest
import numpy as np
import cv2
from src.core.vectorization import StrokeTokenizer

class TestStrokeTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = StrokeTokenizer()

    def test_dot_detection(self):
        # Create a small dot
        image = np.zeros((100, 100), dtype=np.uint8)
        image[50:54, 50:54] = 255 # 4x4 dot

        cleaned, tokens = self.tokenizer.tokenize(image, square_id=1)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, 'Dot')

    def test_line_detection(self):
        # Create a line
        image = np.zeros((100, 100), dtype=np.uint8)
        image[20:80, 50:55] = 255 # 60x5 line

        cleaned, tokens = self.tokenizer.tokenize(image, square_id=2)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, 'Line')

    def test_noise_filtering_isolated_dot(self):
        # Create an isolated dot
        image = np.zeros((100, 100), dtype=np.uint8)
        image[10:13, 10:13] = 255 # Isolated small dot

        # In Square 2, isolated dots should be noise
        cleaned, tokens = self.tokenizer.tokenize(image, square_id=2)

        # Should return as valid token in the list, but cleaned image might depend on implementation.
        # Wait, my implementation returns "valid_tokens" which EXCLUDES noise.
        # Let's check the implementation again.
        # "if is_noise: noise_tokens.append(token)" and "return cleaned_image, valid_tokens"

        self.assertEqual(len(tokens), 0) # Should be filtered out

    def test_noise_filtering_square_1(self):
        # Create an isolated dot in Square 1
        image = np.zeros((100, 100), dtype=np.uint8)
        image[10:13, 10:13] = 255 # Isolated small dot

        # In Square 1, dots are significant
        cleaned, tokens = self.tokenizer.tokenize(image, square_id=1)

        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, 'Dot')

    def test_noise_filtering_near_stroke(self):
        # Create a dot near a line
        image = np.zeros((100, 100), dtype=np.uint8)
        image[20:80, 50:55] = 255 # Line
        image[25:28, 58:61] = 255 # Dot close to line

        cleaned, tokens = self.tokenizer.tokenize(image, square_id=2)

        # Should keep both
        self.assertEqual(len(tokens), 2)

if __name__ == '__main__':
    unittest.main()
