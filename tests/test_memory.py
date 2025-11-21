import unittest
import numpy as np
import os
import cv2
from src.features.memory import SymbolRegistry

class TestSymbolRegistry(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_symbol_db.json"
        self.registry = SymbolRegistry(db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_register_and_match(self):
        # Create a dummy image (e.g., a circle)
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (50, 50), 30, 255, -1)

        self.registry.register_symbol(image, "Circle", ["geometric", "round"])

        # Create a similar image (slightly shifted circle)
        query_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(query_image, (52, 52), 30, 255, -1)

        match = self.registry.find_match(query_image, threshold=0.8)

        self.assertIsNotNone(match)
        self.assertEqual(match["label"], "Circle")
        self.assertIn("geometric", match["tags"])

    def test_no_match(self):
        # Register a circle
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (50, 50), 30, 255, -1)
        self.registry.register_symbol(image, "Circle", [])

        # Query with a square
        query_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(query_image, (20, 20), (80, 80), 255, -1)

        match = self.registry.find_match(query_image, threshold=0.95)
        self.assertIsNone(match)

if __name__ == '__main__':
    unittest.main()
