import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class StrokeToken:
    type: str  # 'Dot', 'Line', 'Noise'
    stats: dict  # area, width, height, centroid, etc.
    mask: np.ndarray # The component mask (binary)

class StrokeTokenizer:
    """
    Tokenizes a binary image into fundamental stroke elements (Dots, Lines)
    and filters out noise based on context.
    """

    DOT_AREA_THRESHOLD: int = 100
    ASPECT_RATIO_THRESHOLD: float = 3.0
    ISOLATION_DISTANCE_THRESHOLD: float = 50.0

    def __init__(self, dot_area_threshold: int = 100, aspect_ratio_threshold: float = 3.0, isolation_distance_threshold: float = 50.0):
        self.dot_area_threshold = dot_area_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.isolation_distance_threshold = isolation_distance_threshold

    def _classify_blob(self, stats: dict) -> str:
        """
        Classifies a blob based on its geometric properties.
        """
        w, h = stats['width'], stats['height']
        area = stats['area']

        # Aspect ratio
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

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
        Checks if a token is isolated from other meaningful tokens.
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
        Analyzes the binary image, extracts tokens, and filters noise.
        Returns the cleaned image (noise removed) and the list of valid tokens.
        """
        # Ensure binary
        if len(binary_image.shape) == 3:
             binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

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
