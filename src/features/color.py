import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Tuple, List

class ColorAnalyzer:
    """
    Analyzes color characteristics of an image for psychological interpretation.
    """

    def __init__(self):
        pass

    def analyze_palette(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyzes the color palette of the image.
        Converts to HSV to calculate Warm vs. Cold and Dark vs. Light ratios.

        Args:
            image: Input image (BGR format as loaded by OpenCV).

        Returns:
            Dictionary containing ratios:
            - warm_cold_ratio: Ratio of warm pixels to cold pixels.
            - light_dark_ratio: Ratio of light pixels to dark pixels.
        """
        if image is None:
            raise ValueError("Image cannot be None")

        if len(image.shape) == 2: # Grayscale
             return {
                "warm_cold_ratio": 0.0,
                "light_dark_ratio": self._calculate_light_dark_ratio(image)
            }

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        # Define Hue ranges for Warm and Cold colors (0-180 in OpenCV)
        # Warm: Red (0-15, 165-180), Orange (15-30), Yellow (30-45) roughly
        # Cold: Green (45-75), Cyan (75-105), Blue (105-135), Violet (135-165) roughly

        # Note: Precise psychological color mapping can vary, using standard approximations.
        # Warm: [0, 45] U [165, 180]
        # Cold: [45, 165]

        warm_mask = ((h >= 0) & (h < 45)) | ((h >= 165) & (h <= 180))
        cold_mask = (h >= 45) & (h < 165)

        # Filter by saturation to avoid counting grays as colors
        # Typically low saturation is achromatic. Let's say S > 20 (approx 8% of 255)
        saturation_mask = s > 20

        warm_pixels = np.sum(warm_mask & saturation_mask)
        cold_pixels = np.sum(cold_mask & saturation_mask)

        warm_cold_ratio = 0.0
        if cold_pixels > 0:
            warm_cold_ratio = warm_pixels / cold_pixels
        elif warm_pixels > 0:
            warm_cold_ratio = float('inf') # Or some max value

        # Light vs Dark
        # Using V channel.
        # Light: V > 127
        # Dark: V <= 127

        light_dark_ratio = self._calculate_light_dark_ratio(v)

        return {
            "warm_cold_ratio": warm_cold_ratio,
            "light_dark_ratio": light_dark_ratio,
            "warm_pixel_count": int(warm_pixels),
            "cold_pixel_count": int(cold_pixels)
        }

    def _calculate_light_dark_ratio(self, channel: np.ndarray) -> float:
        light_mask = channel > 127
        dark_mask = channel <= 127

        light_pixels = np.sum(light_mask)
        dark_pixels = np.sum(dark_mask)

        if dark_pixels > 0:
            return light_pixels / dark_pixels
        elif light_pixels > 0:
            return float('inf')
        return 0.0

    def detect_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
        """
        Uses K-Means clustering to find the dominant colors in the image.

        Args:
            image: Input image (BGR format).
            k: Number of dominant colors to find.

        Returns:
            List of (R, G, B) tuples representing the dominant colors.
        """
        if image is None:
            raise ValueError("Image cannot be None")

        # Reshape the image to a list of pixels
        if len(image.shape) == 3:
            pixels = image.reshape(-1, 3)
        else:
            # If grayscale, treat as 3 channels for consistency or handle separately
            pixels = image.reshape(-1, 1)
            # For now assuming color analysis primarily for color images,
            # but if grayscale, KMeans will return gray values.
            # Let's convert grayscale to BGR for consistency if needed,
            # or just handle the output.
            # Strategy: If grayscale, duplicate channels to make it compatible with BGR return type expected usually.
            # But user asked for "colors".
            # Let's just run kmeans on whatever we have.

        # Convert to float32 for KMeans
        pixels = np.float32(pixels)

        if len(pixels) < k:
             # Not enough pixels
             return []

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_

        # Convert back to uint8
        colors = colors.astype(int)

        # If BGR input, we usually want to return RGB for general usage or keep BGR.
        # Let's assume we return RGB as it's standard for web/display.
        dominant_colors = []
        for color in colors:
            if len(color) == 3:
                # BGR to RGB
                rgb = (int(color[2]), int(color[1]), int(color[0]))
                dominant_colors.append(rgb)
            elif len(color) == 1:
                # Grayscale
                val = int(color[0])
                dominant_colors.append((val, val, val))

        return dominant_colors
