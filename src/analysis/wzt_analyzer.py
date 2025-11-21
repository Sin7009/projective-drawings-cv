import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import os

from src.core.image_processing import ImagePreprocessor
from src.core.vectorization import StrokeTokenizer
from src.features.memory import SymbolRegistry
# Ensure FeatureExtractor is imported for analysis
from src.features.extraction import FeatureExtractor

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

            # 2c. Standard Analysis (Specific CV Logic)
            # We perform this even if semantic match is found, to get the metrics
            features = self._perform_standard_analysis(cleaned_image, square_id)
            square_result["features"] = features

            results[square_id] = square_result

        return results

    def _perform_standard_analysis(self, image: np.ndarray, square_id: int) -> Dict[str, Any]:
        """
        Dispatches to specific feature extraction logic based on square_id.
        Delegates to FeatureExtractor for squares 1, 3, 4, and 5.
        """
        extracted_features: Dict[str, Any] = {}

        if square_id == 1:
            extracted_features = FeatureExtractor.extract_square_1_features(image)
        elif square_id == 3:
            extracted_features = FeatureExtractor.extract_square_3_features(image)
        elif square_id == 4:
            extracted_features = FeatureExtractor.extract_square_4_features(image)
        elif square_id == 5:
            extracted_features = FeatureExtractor.extract_square_5_features(image)
        else:
            # Default or TODO for other squares
            extracted_features = {
                "status": "pending_implementation",
                "square_id": square_id
            }

        return extracted_features
