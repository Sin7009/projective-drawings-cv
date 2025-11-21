import cv2
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any, List

class FeatureExtractor:
    """
    Static methods for extracting specific Wartegg features based on the square ID.
    """

    @staticmethod
    def extract_square_1_features(image: np.ndarray) -> Dict[str, float]:
        """
        Square 1 (Ego/Point):
        - Find the centroid of all drawn pixels.
        - Calculate displacement_vector: (drawing_center_x - image_center_x, drawing_center_y - image_center_y).
        - Return scalar distance and vector angle.
        """
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
    def extract_square_3_features(image: np.ndarray) -> Dict[str, float]:
        """
        Square 3 (Ambition/Lines):
        - Skeletonize the image (reduce lines to 1px width).
        - Get coordinates of all lit pixels.
        - Perform Linear Regression (scipy.stats.linregress) on these points.
        - Return slope (m) and intercept (b). High positive slope = high ambition.
        """
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

        # Note: In image coordinates, Y increases downwards.
        # If "High positive slope = high ambition" refers to visual "uphill",
        # in Cartesian coordinates (y up), slope > 0.
        # In image coordinates (y down), visual uphill means y decreases as x increases -> slope < 0.
        # The requirement says "High positive slope = high ambition".
        # Usually in CV for graphs, we might invert Y.
        # Assuming standard mathematical slope interpretation on the visual representation:
        # Visual "Uphill" (/): x increases, visual y increases (image y decreases).
        # Let's just return the calculated slope for now, but keep in mind the coordinate system.
        # If the user means visually ascending lines, that corresponds to negative slope in image coordinates (0,0 at top-left).
        # BUT, standard linregress on (x, y_image) gives dy_image/dx.
        # A line from (0, 100) to (100, 0) [visually uphill] has slope (0-100)/(100-0) = -1.
        # A line from (0, 0) to (100, 100) [visually downhill] has slope 1.
        # NOTE: We return the raw mathematical slope in image coordinates (y-down).
        # Downstream interpretation logic must account for this:
        # - High Ambition (Visual Uphill) corresponds to Negative Slope values.
        # - Low Ambition (Visual Downhill) corresponds to Positive Slope values.

        return {
            "slope": float(slope), # In image coordinates
            "intercept": float(intercept),
            "r_value": float(r_value)
        }

    @staticmethod
    def extract_square_4_features(image: np.ndarray) -> Dict[str, Any]:
        """
        Square 4 (Anxiety/Darkness):
        - Calculate pixel_density: (count of black pixels / total pixels).
          Wait, the input image is likely binarized.
          The requirement says "count of black pixels".
          If our binarization makes drawing white on black background (standard for processing),
          then we should count white pixels as "drawn pixels".
          If the requirement literally means "darkness" in the original drawing,
          then "black pixels" (ink) corresponds to "white pixels" in our inverted binary map.
          Let's assume 'image' is white-on-black (drawing is white).
        - Detect if the pre-printed black square was heavily shaded over (check ROI around the stimulus).
        """
        h, w = image.shape
        total_pixels = h * w
        drawn_pixels = np.count_nonzero(image)

        pixel_density = drawn_pixels / total_pixels if total_pixels > 0 else 0.0

        # Detect shading over stimulus
        # Square 4 stimulus is a small black square.
        # In the original Wartegg grid, Square 4 has a small black square in the top right?
        # Actually usually it's Top Right.
        # Let's assume we need to check a specific ROI.
        # Since I don't have the exact coordinates of the stimulus relative to the cell,
        # I'll assume it's in the top-right quadrant based on standard WZT Square 4.
        # But wait, the prompt doesn't specify the location.
        # "check ROI around the stimulus".
        # I will define a heuristic ROI. Standard Square 4 stimulus is usually near top-right.
        # Let's define a region in top right.

        # Using the blank WZT description from memory/online sources:
        # Square 4: Small black square in the upper right corner.
        roi_x_start = int(w * 0.7)
        roi_y_end = int(h * 0.3)
        roi = image[0:roi_y_end, roi_x_start:w]

        roi_density = np.count_nonzero(roi) / (roi.size) if roi.size > 0 else 0.0

        # Heuristic: if high density in that area, it's shaded over.
        heavily_shaded = roi_density > 0.5

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
        - Use cv2.HoughLinesP to find line segments.
        - Check if lines intersect or connect the two stimulus strokes.
        - Return intersection_count and average_line_length.
        """
        # HoughLinesP requires 8-bit single channel
        lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=10)

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

        # Check intersections
        # A naive approach checks intersection between all pairs of detected lines.
        # The requirement says "Check if lines intersect or connect the two stimulus strokes."
        # Since I don't know the exact position of stimulus strokes in the image coordinates
        # (unless I find them, but they are part of the image now),
        # I will count intersections between the drawn lines themselves as a proxy for "intersection count" complexity
        # OR I need to identify which lines are stimulus.
        # Without a reference blank or stimulus separation, it's hard to know which pixels are stimulus.
        # Assuming the prompt "intersection_count" refers to intersections found among the detected lines.

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
