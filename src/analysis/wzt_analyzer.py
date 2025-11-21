import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import os

from src.core.image_processing import ImagePreprocessor
from src.core.vectorization import StrokeTokenizer
from src.features.memory import SymbolRegistry

class WarteggAnalyzer:
    """
    Analyzes Wartegg Zeichentest (WZT) drawings.
    Integrates preprocessing, stroke tokenization, and semantic memory lookup.
    """

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.tokenizer = StrokeTokenizer()
        self.memory = SymbolRegistry()
        # Ensure we have a DB file or it will start empty

    def process_and_analyze(self, image_path: str) -> Dict[int, Dict[str, Any]]:
        """
        Main pipeline:
        1. Preprocess image (deskew, crop squares).
        2. For each square:
           a. Tokenize and clean (remove noise).
           b. Check Semantic Memory for known symbols.
           c. (If no match) Perform standard analysis (placeholder).
        """

        # 1. Preprocess
        try:
            squares = self.preprocessor.process(image_path)
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return {}

        results = {}

        for square_id, image in squares.items():
            # 2a. Tokenize and Clean
            # Assuming image is binarized white-on-black from preprocessor
            cleaned_image, tokens = self.tokenizer.tokenize(image, square_id)

            square_result = {
                "token_count": len(tokens),
                "tokens": [t.type for t in tokens],
                "semantic_match": None,
                "analysis_mode": "Standard"
            }

            # 2b. Semantic Memory Lookup
            # Only check if there is significant content
            if len(tokens) > 0:
                match = self.memory.find_match(cleaned_image)
                if match:
                    square_result["semantic_match"] = match
                    square_result["analysis_mode"] = "Semantic"

            # 2c. Standard Analysis (Placeholder for future implementation)
            if square_result["analysis_mode"] == "Standard":
                square_result["features"] = self._perform_standard_analysis(cleaned_image, square_id)

            results[square_id] = square_result

        return results

    def _perform_standard_analysis(self, image: np.ndarray, square_id: int) -> Dict[str, Any]:
        """
        Placeholder for specific per-square analysis logic.
        """
        # Real implementation would go here (e.g., curve analysis for Square 2)
        return {
            "status": "analyzed",
            "square_id": square_id
        }
