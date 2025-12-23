import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import os

from src.features.processing import ImagePreprocessor
from src.core.vectorization import StrokeTokenizer
from src.features.memory import SymbolRegistry
from src.features.extraction import FeatureExtractor

# Constants
MIN_TOKEN_COUNT_FOR_SEMANTIC_CHECK = 0


class WarteggAnalyzer:
    """
    Analyzes Wartegg Zeichentest (WZT) drawings using a multi-stage pipeline.
    
    Integrates:
    - Image preprocessing and grid extraction
    - Stroke tokenization and noise filtering
    - Semantic memory lookup for known symbols
    - Square-specific feature extraction
    """

    def __init__(self):
        """Initialize the analyzer with required components."""
        self.preprocessor = ImagePreprocessor()
        self.tokenizer = StrokeTokenizer()
        self.memory = SymbolRegistry()

    def process_and_analyze(self, image_path: str) -> Dict[int, Dict[str, Any]]:
        """
        Main analysis pipeline for Wartegg drawings.
        
        Pipeline stages:
        1. Preprocess image (deskew, extract squares)
        2. For each square:
           a. Tokenize and clean (remove noise)
           b. Check semantic memory for known symbols
           c. Perform square-specific analysis
           
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary mapping square IDs (1-8) to their analysis results
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If preprocessing fails
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
            if len(tokens) > MIN_TOKEN_COUNT_FOR_SEMANTIC_CHECK:
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
        Dispatch to specific feature extraction logic based on square ID.
        
        Each square has psychologically meaningful stimuli that elicit specific features:
        - Square 1: Ego/Self (centroid displacement)
        - Square 2: Empathy (smoothness of curves)
        - Square 3: Ambition (line slopes)
        - Square 4: Anxiety (shading density)
        - Square 5: Aggression (line intersections)
        - Square 6: Integration (closed shapes)
        - Square 7: Sensitivity (dot connections)
        - Square 8: Protection (positioning under arc)
        
        Args:
            image: Binary image of the square
            square_id: Square identifier (1-8)
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}

        if square_id == 1:
            features = FeatureExtractor.extract_square_1_features(image)
        elif square_id == 2:
            features = FeatureExtractor.extract_square_2_features(image)
        elif square_id == 3:
            features = FeatureExtractor.extract_square_3_features(image)
        elif square_id == 4:
            features = FeatureExtractor.extract_square_4_features(image)
        elif square_id == 5:
            features = FeatureExtractor.extract_square_5_features(image)
        elif square_id == 6:
            features = FeatureExtractor.extract_square_6_features(image)
        elif square_id == 7:
            features = FeatureExtractor.extract_square_7_features(image)
        elif square_id == 8:
            features = FeatureExtractor.extract_square_8_features(image)
        else:
            features = {
                "status": "unknown_square_id",
                "square_id": square_id
            }

        return features
