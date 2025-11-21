import numpy as np
import json
import os
import cv2
from typing import List, Dict, Optional, Tuple

class SymbolRegistry:
    """
    A simple embedding store for registering and recognizing recurring symbols.
    """

    def __init__(self, db_path: str = "symbol_db.json"):
        self.db_path = db_path
        self.registry = self._load_db()

    def _load_db(self) -> List[Dict]:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def _save_db(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def _compute_vector(self, image: np.ndarray) -> List[float]:
        """
        Computes a feature vector for the image focusing on shape descriptors.
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Find largest contour
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return [0.0] * 11

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area == 0:
             return [0.0] * 11

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

        # 2. Hu Moments (Log transformed)
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

        # Normalize Hu moments to have roughly unit range?
        # Hu moments logs are typically around -2 to -10 or so.
        # We won't normalize heavily, just use them.

        descriptors = [solidity, extent, circularity, aspect_ratio]

        vector = np.concatenate([
            np.array(descriptors),
            hu_logs
        ])

        # Clean NaNs or Infs
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

        return vector.tolist()

    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.linalg.norm(v1 - v2))

    def register_symbol(self, image: np.ndarray, label: str, tags: List[str]):
        """
        Registers a new symbol in the database.
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
        Finds a matching symbol in the database.

        Note: The threshold here is interpreted as a SIMILARITY threshold.
        Since we use Euclidean distance, we convert distance to similarity:
        similarity = 1 / (1 + distance)
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
