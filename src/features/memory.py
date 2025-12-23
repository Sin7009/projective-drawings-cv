import numpy as np
import json
import os
import cv2
from typing import List, Dict, Optional, Tuple

# Constants for symbol registry
DEFAULT_DB_PATH = "symbol_db.json"
FEATURE_VECTOR_LENGTH = 11
MIN_CONTOUR_AREA = 0


class SymbolRegistry:
    """
    A simple embedding store for registering and recognizing recurring symbols.
    Uses shape descriptors and Hu moments for rotation-invariant symbol matching.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Initialize the symbol registry.
        
        Args:
            db_path: Path to the JSON database file
        """
        self.db_path = db_path
        self.registry = self._load_db()

    def _load_db(self) -> List[Dict]:
        """
        Load the symbol database from disk.
        
        Returns:
            List of symbol entries or empty list if file doesn't exist/is invalid
        """
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load symbol database: {e}")
                return []
        return []

    def _save_db(self) -> None:
        """
        Save the symbol database to disk.
        
        Raises:
            IOError: If unable to write to the database file
        """
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except IOError as e:
            print(f"Error: Failed to save symbol database: {e}")
            raise

    def _compute_vector(self, image: np.ndarray) -> List[float]:
        """
        Compute a feature vector for an image using shape descriptors and Hu moments.
        
        The vector combines:
        - Geometric descriptors: solidity, extent, circularity, aspect ratio
        - Hu moments: 7 rotation-invariant moment features (log transformed)
        
        Args:
            image: Binary image to compute features for
            
        Returns:
            11-element feature vector (4 descriptors + 7 Hu moments)
            
        Raises:
            ValueError: If image is None or empty
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
            
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Find largest contour
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return [0.0] * FEATURE_VECTOR_LENGTH

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area <= MIN_CONTOUR_AREA:
            return [0.0] * FEATURE_VECTOR_LENGTH

        # 1. Shape Descriptors

        # Convex Hull & Solidity
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Bounding Rect & Extent
        x, y, w, h = cv2.boundingRect(c)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0

        # Aspect Ratio
        aspect_ratio = float(w) / h if h > 0 else 0

        # Circularity
        perimeter = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

        # 2. Hu Moments (Log transformed for scale invariance)
        moments = cv2.moments(c)
        hu = cv2.HuMoments(moments).flatten()

        hu_logs = []
        for val in hu:
            if val == 0:
                hu_logs.append(0.0)
            else:
                # Log transform to handle scale
                hu_logs.append(np.sign(val) * np.log10(np.abs(val)))

        hu_logs = np.array(hu_logs)

        descriptors = [solidity, extent, circularity, aspect_ratio]

        vector = np.concatenate([
            np.array(descriptors),
            hu_logs
        ])

        # Clean NaNs or Infs
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

        return vector.tolist()

    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate Euclidean distance between two vectors.
        
        Args:
            vec1: First feature vector
            vec2: Second feature vector
            
        Returns:
            Euclidean distance
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.linalg.norm(v1 - v2))

    def register_symbol(self, image: np.ndarray, label: str, tags: List[str]) -> None:
        """
        Register a new symbol in the database.
        
        Args:
            image: Binary image of the symbol
            label: Text label for the symbol
            tags: List of descriptive tags
            
        Raises:
            ValueError: If image is invalid
            IOError: If unable to save to database
        """
        vector = self._compute_vector(image)
        entry = {
            "label": label,
            "tags": tags,
            "vector": vector
        }
        self.registry.append(entry)
        self._save_db()

    def find_match(self, image: np.ndarray, threshold: float = 0.85) -> Optional[Dict]:
        """
        Find a matching symbol in the database using similarity matching.
        
        Uses Euclidean distance converted to similarity: similarity = 1 / (1 + distance)
        
        Args:
            image: Binary image to match
            threshold: Minimum similarity threshold (0-1), higher is more strict
            
        Returns:
            Dictionary with match information (label, tags, score) or None if no match
            
        Raises:
            ValueError: If image is invalid
        """
        query_vector = self._compute_vector(image)

        best_match = None
        best_similarity = -1.0

        for entry in self.registry:
            dist = self._euclidean_distance(query_vector, entry["vector"])
            similarity = 1.0 / (1.0 + dist)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        if best_similarity >= threshold and best_match:
            return {
                "label": best_match["label"],
                "tags": best_match["tags"],
                "score": best_similarity
            }

        return None
