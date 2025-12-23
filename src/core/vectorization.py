import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from src.config import config

# Constants for stroke tokenization
DEFAULT_DOT_AREA_THRESHOLD = 100
DEFAULT_ASPECT_RATIO_THRESHOLD = 3.0
DEFAULT_ISOLATION_DISTANCE = 50.0
MIN_DIMENSION_EPSILON = 1e-5
CONNECTED_COMPONENTS_CONNECTIVITY = 8


@dataclass
class StrokeToken:
    """
    Represents a fundamental stroke element in a drawing.
    
    Attributes:
        type: Classification of the stroke ('Dot', 'Line', 'Noise')
        stats: Dictionary containing geometric properties (area, width, height, centroid)
        mask: Binary image mask of the component
    """
    type: str  # 'Dot', 'Line', 'Noise'
    stats: dict  # area, width, height, centroid, etc.
    mask: np.ndarray  # The component mask (binary)

class StrokeTokenizer:
    """
    Tokenizes a binary image into fundamental stroke elements (Dots, Lines)
    and filters out noise based on geometric context.
    
    This class performs intelligent segmentation of children's drawings,
    distinguishing meaningful strokes from scanning artifacts.
    """

    def __init__(
        self, 
        dot_area_threshold: Optional[int] = None, 
        aspect_ratio_threshold: Optional[float] = None, 
        isolation_distance_threshold: float = DEFAULT_ISOLATION_DISTANCE
    ):
        """
        Initialize the stroke tokenizer.
        
        Args:
            dot_area_threshold: Maximum area for classifying as a dot (uses config default if None)
            aspect_ratio_threshold: Minimum aspect ratio for line classification (uses config default if None)
            isolation_distance_threshold: Distance threshold for isolation detection
        """
        self.dot_area_threshold = dot_area_threshold or config.get(
            'vectorization.dot_max_area', DEFAULT_DOT_AREA_THRESHOLD
        )
        self.aspect_ratio_threshold = aspect_ratio_threshold or config.get(
            'vectorization.line_aspect_ratio', DEFAULT_ASPECT_RATIO_THRESHOLD
        )
        self.isolation_distance_threshold = isolation_distance_threshold

    def _classify_blob(self, stats: dict) -> str:
        """
        Classify a blob based on its geometric properties.
        
        Args:
            stats: Dictionary containing blob statistics (width, height, area)
            
        Returns:
            Classification string: 'Dot' or 'Line'
        """
        w, h = stats['width'], stats['height']
        area = stats['area']

        # Aspect ratio with safe division
        aspect_ratio = max(w, h) / (min(w, h) + MIN_DIMENSION_EPSILON)

        # Dot criteria: Compact shape, small area
        if aspect_ratio < self.aspect_ratio_threshold and area < self.dot_area_threshold:
            return 'Dot'

        # Line criteria: Elongated shape
        if aspect_ratio >= self.aspect_ratio_threshold:
            return 'Line'

        # Default fallback
        return 'Line'

    def _is_isolated(self, token: StrokeToken, all_tokens: List[StrokeToken]) -> bool:
        """
        Check if a token is spatially isolated from other meaningful tokens.
        
        Args:
            token: Token to check for isolation
            all_tokens: List of all tokens to check against
            
        Returns:
            True if token is isolated, False otherwise
        """
        c1 = np.array(token.stats['centroid'])

        for other in all_tokens:
            if other is token:
                continue

            c2 = np.array(other.stats['centroid'])
            dist = np.linalg.norm(c1 - c2)

            if dist < self.isolation_distance_threshold:
                return False

        return True

    def tokenize(self, binary_image: np.ndarray, square_id: int) -> Tuple[np.ndarray, List[StrokeToken]]:
        """
        Analyze the binary image, extract tokens, and filter noise.
        
        Args:
            binary_image: Input binary image
            square_id: ID of the Wartegg square (1-8) for context-specific filtering
            
        Returns:
            Tuple of (cleaned_image, valid_tokens)
            - cleaned_image: Binary image with noise removed
            - valid_tokens: List of valid StrokeToken objects
            
        Raises:
            ValueError: If binary_image is None or empty
        """
        if binary_image is None or binary_image.size == 0:
            raise ValueError("Binary image cannot be None or empty")
            
        # Ensure binary
        if len(binary_image.shape) == 3:
            binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=CONNECTED_COMPONENTS_CONNECTIVITY
        )

        tokens = []

        # Skip label 0 (background)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]

            # Extract individual mask
            mask = (labels == i).astype(np.uint8) * 255

            token_stats = {
                'x': x, 'y': y, 'width': w, 'height': h, 'area': area,
                'centroid': (cx, cy)
            }

            token_type = self._classify_blob(token_stats)

            token = StrokeToken(type=token_type, stats=token_stats, mask=mask)
            tokens.append(token)

        # Apply Noise Filter logic
        valid_tokens = []
        noise_tokens = []

        # First pass: Separate potential dots
        for token in tokens:
            is_noise = False

            if token.type == 'Dot':
                # Logic: If a 'Dot' is isolated AND square_id != 1 and != 7 -> Noise
                # Squares 1 and 7 are expected to have dots as part of the drawing
                if square_id not in [1, 7]:
                    if self._is_isolated(token, tokens):
                        is_noise = True

            if is_noise:
                token.type = 'Noise/Artifact'
                noise_tokens.append(token)
            else:
                valid_tokens.append(token)

        # Reconstruct cleaned image
        cleaned_image = np.zeros_like(binary_image)
        for token in valid_tokens:
            cleaned_image = cv2.bitwise_or(cleaned_image, token.mask)

        return cleaned_image, valid_tokens
