import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.analysis.wzt_analyzer import WarteggAnalyzer
from src.core.vectorization import StrokeToken

class TestWarteggAnalyzer(unittest.TestCase):
    @patch('src.analysis.wzt_analyzer.ImagePreprocessor')
    @patch('src.analysis.wzt_analyzer.StrokeTokenizer')
    @patch('src.analysis.wzt_analyzer.SymbolRegistry')
    def test_process_and_analyze(self, MockRegistry, MockTokenizer, MockPreprocessor):
        # Setup mocks
        preprocessor = MockPreprocessor.return_value
        tokenizer = MockTokenizer.return_value
        registry = MockRegistry.return_value

        analyzer = WarteggAnalyzer()

        # Mock preprocessor output
        mock_squares = {
            1: np.zeros((100, 100), dtype=np.uint8),
            2: np.zeros((100, 100), dtype=np.uint8)
        }
        preprocessor.process.return_value = mock_squares

        # Mock tokenizer output
        # Square 1: No tokens (blank)
        # Square 2: Some tokens
        def tokenize_side_effect(image, square_id):
            if square_id == 1:
                return image, []
            else:
                token = StrokeToken(type='Line', stats={}, mask=image)
                return image, [token]

        tokenizer.tokenize.side_effect = tokenize_side_effect

        # Mock registry output
        # Square 2: Match found
        registry.find_match.return_value = {"label": "Tree", "tags": ["nature"], "score": 0.9}

        results = analyzer.process_and_analyze("dummy_path.jpg")

        # Check Square 1
        self.assertEqual(len(results[1]['tokens']), 0)
        self.assertIsNone(results[1]['semantic_match'])
        self.assertEqual(results[1]['analysis_mode'], 'Standard')

        # Check Square 2
        self.assertEqual(len(results[2]['tokens']), 1)
        self.assertIsNotNone(results[2]['semantic_match'])
        self.assertEqual(results[2]['semantic_match']['label'], "Tree")
        self.assertEqual(results[2]['analysis_mode'], 'Semantic')

if __name__ == '__main__':
    unittest.main()
