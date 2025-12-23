import cv2
import numpy as np
from scipy import stats
from scipy.stats import entropy
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
from loguru import logger
import skimage.feature as ft

# Constants for image processing
BACKGROUND_BLUR_KERNEL_SIZE = 101
INK_DETECTION_THRESHOLD = 10
GLCM_DISTANCE = 1
GLCM_LEVELS = 256
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
HOUGH_THRESHOLD = 20
HOUGH_MIN_LINE_LENGTH = 10
HOUGH_MAX_LINE_GAP = 10
ROI_DENSITY_THRESHOLD = 0.5
CENTROID_Y_THRESHOLD = 0.4
MAX_CONTOURS_FOR_CONNECTION = 4
MIN_DRAWN_PIXELS = 10


class FeatureExtractor:
    """
    Static methods for extracting features from drawings.
    Includes specific Wartegg square features and global texture/pressure metrics.
    """

    @staticmethod
    def extract_global_features(image_path: Path) -> Optional[Dict[str, Any]]:
        """
        Extracts global features (Pressure, Entropy, Texture) from a raw image.

        Algorithmic Improvements:
        - Robust Pressure: Background subtraction (GaussianBlur -> Divide).
        - Rotation Invariant GLCM: Average over 4 angles.
        - Safety: Handles missing files/None images.
        """
        # Safety check: path validity
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None

        # Load image (Read as Color to preserve potential color info, though we mostly use gray)
        # Using cv2.IMREAD_COLOR for general compatibility, then converting.
        img = cv2.imread(str(image_path))

        # Safety check: valid image
        if img is None:
            logger.error(f"Failed to load image (cv2 returned None): {image_path}")
            return None

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # --- 1. Robust Pressure Calculation ---
            # "Apply GaussianBlur to estimate background. Divide original image by background."

            # Estimate background (illumination)
            # Kernel size needs to be large enough to blur out the strokes but keep the background trend
            # If image is large, kernel should be larger. Assuming standard A4 scan resolution ~2000px width.
            bg_blur = cv2.GaussianBlur(gray, (BACKGROUND_BLUR_KERNEL_SIZE, BACKGROUND_BLUR_KERNEL_SIZE), 0)

            # Avoid division by zero
            bg_blur[bg_blur == 0] = 1

            # Divide: (original / background) * 255. Result is normalized image where bg is white (255)
            # This flattens the lighting.
            normalized = cv2.divide(gray, bg_blur, scale=255)

            # Now, strokes are dark, background is white (near 255).
            # "Mean intensity of the strokes"
            # We need to segment strokes. Simple threshold or check deviations.
            # Inverted: 255 - normalized. Bg becomes 0. Strokes become bright.
            inverted_norm = 255 - normalized

            # Threshold to identify "ink" vs "noise/paper"
            # Since we normalized, paper should be very close to 0 (in inverted).
            ink_mask = inverted_norm > INK_DETECTION_THRESHOLD

            if np.any(ink_mask):
                # Calculate mean of INK pixels only.
                # Higher value = Darker original stroke = Higher Pressure
                mean_pressure = np.mean(inverted_norm[ink_mask])
            else:
                mean_pressure = 0.0

            # --- 2. Entropy (Chaos/Complexity) ---
            # Calculate on the normalized gray image to reduce lighting noise
            hist = cv2.calcHist([normalized], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / hist.sum()
            img_entropy = entropy(hist_norm, base=2)

            # --- 3. Rotation Invariant Texture Analysis (GLCM) ---
            # Compute the GLCM for all 4 angles to achieve rotation invariance
            # skimage expects integer image for GLCM.
            # We use the normalized image to avoid lighting artifacts affecting texture.
            glcm = ft.graycomatrix(
                normalized,
                distances=[GLCM_DISTANCE],
                angles=GLCM_ANGLES,
                levels=GLCM_LEVELS,
                symmetric=True,
                normed=True
            )

            # Properties to extract
            props = ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity', 'ASM']
            texture_feats = {}

            for prop in props:
                # graycoprops returns (num_distances, num_angles)
                val_matrix = ft.graycoprops(glcm, prop)
                # Average over all angles (axis 1) and distances (axis 0 - only 1 distance here)
                # This makes it rotation invariant.
                avg_val = np.mean(val_matrix)
                texture_feats[prop] = round(float(avg_val), 4)

            return {
                "filename": image_path.name,
                "mean_pressure": round(float(mean_pressure), 2),
                "entropy": round(float(img_entropy), 4),
                **texture_feats
            }

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None

    @staticmethod
    def extract_square_1_features(image: np.ndarray) -> Dict[str, float]:
        """
        Square 1 (Ego/Point):
        Analyzes the centroid displacement of drawn content from the image center.
        This can indicate the child's sense of self-positioning.
        
        Args:
            image: Binary image of square 1
            
        Returns:
            Dictionary containing centroid coordinates, displacement distance and angle
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
            
        # Find non-zero pixels
        y_indices, x_indices = np.nonzero(image)

        if len(x_indices) == 0:
            return {
                "centroid_x": 0.0,
                "centroid_y": 0.0,
                "displacement_distance": 0.0,
                "displacement_angle": 0.0
            }

        # Calculate drawing centroid
        centroid_x = np.mean(x_indices)
        centroid_y = np.mean(y_indices)

        # Image center
        h, w = image.shape
        center_x = w / 2.0
        center_y = h / 2.0

        # Displacement vector
        dx = centroid_x - center_x
        dy = centroid_y - center_y

        # Scalar distance
        distance = np.sqrt(dx**2 + dy**2)

        # Vector angle (in degrees)
        angle = np.degrees(np.arctan2(dy, dx))

        return {
            "centroid_x": float(centroid_x),
            "centroid_y": float(centroid_y),
            "displacement_distance": float(distance),
            "displacement_angle": float(angle)
        }

    @staticmethod
    def extract_square_2_features(image: np.ndarray) -> Dict[str, float]:
        """
        Square 2 (Empathy):
        Analyzes the smoothness of curves drawn, which may relate to emotional expression.
        
        Args:
            image: Binary image of square 2
            
        Returns:
            Dictionary containing smoothness ratio and perimeter measurements
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
            
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                "smoothness": 0.0,
                "perimeter": 0.0,
                "hull_perimeter": 0.0
            }

        # Assume largest contour is the main drawing
        main_contour = max(contours, key=cv2.contourArea)

        contour_perimeter = cv2.arcLength(main_contour, True)
        hull = cv2.convexHull(main_contour)
        hull_perimeter = cv2.arcLength(hull, True)

        smoothness = hull_perimeter / contour_perimeter if contour_perimeter > 0 else 0.0

        return {
            "smoothness": float(smoothness),
            "perimeter": float(contour_perimeter),
            "hull_perimeter": float(hull_perimeter)
        }

    @staticmethod
    def extract_square_3_features(image: np.ndarray) -> Dict[str, float]:
        """
        Square 3 (Ambition/Lines):
        Analyzes the slope of drawn lines, which may indicate aspiration levels.
        Uses skeletonization and linear regression.
        
        Args:
            image: Binary image of square 3
            
        Returns:
            Dictionary containing slope, intercept, and correlation coefficient
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
            
        # Skeletonize
        skeleton = cv2.ximgproc.thinning(image)

        # Get coordinates of lit pixels
        y_indices, x_indices = np.nonzero(skeleton)

        if len(x_indices) < 2:
             return {
                "slope": 0.0,
                "intercept": 0.0,
                "r_value": 0.0
            }

        # Perform Linear Regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_indices, y_indices)

        return {
            "slope": float(slope), # In image coordinates
            "intercept": float(intercept),
            "r_value": float(r_value)
        }

    @staticmethod
    def extract_square_4_features(image: np.ndarray) -> Dict[str, Any]:
        """
        Square 4 (Anxiety/Darkness):
        Measures shading density, which may correlate with anxiety levels.
        
        Args:
            image: Binary image of square 4
            
        Returns:
            Dictionary containing pixel density, ROI density, and shading indicators
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
            
        h, w = image.shape
        total_pixels = h * w
        drawn_pixels = np.count_nonzero(image)

        pixel_density = drawn_pixels / total_pixels if total_pixels > 0 else 0.0

        # Detect shading over stimulus
        # Square 4 stimulus is a small black square.
        # Let's assume we need to check a specific ROI.
        roi_x_start = int(w * 0.7)
        roi_y_end = int(h * 0.3)
        roi = image[0:roi_y_end, roi_x_start:w]

        roi_density = np.count_nonzero(roi) / (roi.size) if roi.size > 0 else 0.0

        # Heuristic: if high density in that area, it's shaded over.
        heavily_shaded = roi_density > ROI_DENSITY_THRESHOLD

        return {
            "pixel_density": float(pixel_density),
            "roi_density": float(roi_density),
            "heavily_shaded": bool(heavily_shaded),
            "roi_coords": [0, roi_y_end, roi_x_start, w]  # y1, y2, x1, x2
        }

    @staticmethod
    def extract_square_5_features(image: np.ndarray) -> Dict[str, Any]:
        """
        Square 5 (Aggression/Obstacles):
        Detects line segments and their intersections, which may relate to handling obstacles.
        
        Args:
            image: Binary image of square 5
            
        Returns:
            Dictionary containing line count, intersections, and average line length
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
            
        # HoughLinesP requires 8-bit single channel
        lines = cv2.HoughLinesP(
            image, 1, np.pi / 180, 
            threshold=HOUGH_THRESHOLD, 
            minLineLength=HOUGH_MIN_LINE_LENGTH, 
            maxLineGap=HOUGH_MAX_LINE_GAP
        )

        if lines is None:
            return {
                "intersection_count": 0,
                "average_line_length": 0.0,
                "line_count": 0,
                "lines": []
            }

        total_length = 0.0
        num_lines = len(lines)
        lines_list = []

        # Calculate lengths
        line_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_length += length
            line_segments.append(((x1, y1), (x2, y2)))
            lines_list.append([int(x1), int(y1), int(x2), int(y2)])

        average_length = total_length / num_lines if num_lines > 0 else 0.0

        intersection_count = 0
        for i in range(num_lines):
            for j in range(i + 1, num_lines):
                if FeatureExtractor._lines_intersect(line_segments[i], line_segments[j]):
                    intersection_count += 1

        return {
            "intersection_count": intersection_count,
            "average_line_length": float(average_length),
            "line_count": num_lines,
            "lines": lines_list
        }

    @staticmethod
    def extract_square_6_features(image: np.ndarray) -> Dict[str, Any]:
        """
        Square 6 (Integration):
        Detects closed shapes which may indicate integration and completion tendencies.
        
        Args:
            image: Binary image of square 6
            
        Returns:
            Dictionary indicating presence of closed shapes and contour count
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
            
        # Use RETR_CCOMP to find hierarchy of contours (outer and holes)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                "is_closed_shape": False,
                "num_contours": 0
            }

        # Check if any contour has a child (hierarchy field 2 != -1)
        # Hierarchy: [Next, Previous, First_Child, Parent]
        has_hole = False
        if hierarchy is not None:
            for i in range(len(contours)):
                if hierarchy[0][i][2] != -1:
                    has_hole = True
                    break

        return {
            "is_closed_shape": bool(has_hole),
            "num_contours": len(contours)
        }

    @staticmethod
    def extract_square_7_features(image: np.ndarray) -> Dict[str, Any]:
        """
        Square 7 (Sensitivity):
        Analyzes whether dots are connected, which may indicate emotional expression style.
        
        Args:
            image: Binary image of square 7
            
        Returns:
            Dictionary indicating contour count and connection likelihood
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
            
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)

        # Heuristic for "Connected":
        # Square 7 stimulus has ~8 dots.
        # If connected, they form fewer connected components.
        is_drawn = np.count_nonzero(image) > MIN_DRAWN_PIXELS

        likely_connected = is_drawn and (num_contours <= MAX_CONTOURS_FOR_CONNECTION)

        return {
            "contour_count": num_contours,
            "dots_connected": bool(likely_connected)
        }

    @staticmethod
    def extract_square_8_features(image: np.ndarray) -> Dict[str, Any]:
        """
        Square 8 (Protection):
        Analyzes whether drawing is positioned under the arc, indicating need for protection.
        
        Args:
            image: Binary image of square 8
            
        Returns:
            Dictionary with centroid position and under-arc indicator
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be None or empty")
            
        y_indices, x_indices = np.nonzero(image)

        if len(y_indices) == 0:
             return {
                 "centroid_y_norm": 0.0,
                 "is_under_arc": False
             }

        h, w = image.shape
        centroid_y = np.mean(y_indices)

        # Normalize centroid Y (0 is top, 1 is bottom)
        norm_centroid_y = centroid_y / h

        # Threshold: if centroid is in the lower part, it's likely under the arc.
        # The arc itself is usually near the top.
        is_under_arc = norm_centroid_y > CENTROID_Y_THRESHOLD

        return {
            "centroid_y_norm": float(norm_centroid_y),
            "is_under_arc": bool(is_under_arc)
        }

    @staticmethod
    def _lines_intersect(line1, line2) -> bool:
        """
        Check if two line segments intersect.
        line1: ((x1, y1), (x2, y2))
        """
        p1, q1 = line1
        p2, q2 = line2

        # Using orientation method
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0: return 0  # colinear
            return 1 if val > 0 else 2 # clock or counterclock

        def on_segment(p, q, r):
            if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
                return True
            return False

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p2, q1): return True
        if o2 == 0 and on_segment(p1, q2, q1): return True
        if o3 == 0 and on_segment(p2, p1, q2): return True
        if o4 == 0 and on_segment(p2, q1, q2): return True

        return False
