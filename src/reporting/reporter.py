import csv
import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.features.processing import ImagePreprocessor

class Reporter:
    """
    Handles reporting and visualization of Wartegg analysis results.
    """

    def __init__(self):
        self.preprocessor = ImagePreprocessor()

    def export_to_csv(self, analysis_results: Dict[int, Dict[str, Any]], filename: str, output_csv: str = "results.csv"):
        """
        Exports analysis results to a CSV file.

        Args:
            analysis_results: The dictionary returned by WarteggAnalyzer.
            filename: The name of the image file analyzed.
            output_csv: Path to the output CSV file.
        """
        # Flatten the results
        row_data = {"Filename": filename}

        # Check for top-level anxiety prediction (if passed in a wrapper dict,
        # but here we assume analysis_results is the direct output of analyzer,
        # so we might need to check if the caller injects it or if it's in a specific square).
        # Based on current analyzer, it returns {square_id: {...}}.
        # If Anxiety_Prediction is expected, it might be passed separately or injected.
        # For now, we'll check if it's in the dict under a special key or just default to N/A.
        row_data["Anxiety_Prediction"] = analysis_results.get("Anxiety_Prediction", "N/A")

        for sq_id, data in analysis_results.items():
            if not isinstance(sq_id, int):
                continue # Skip non-square keys if any

            features = data.get("features", {})
            prefix = f"Sq{sq_id}"

            for feature_name, value in features.items():
                # Skip complex structures like lines list or ROI coords for CSV
                if isinstance(value, (list, dict, np.ndarray)):
                    continue

                # Map specific names to requested names if needed, or just use consistent naming
                # Request: Sq1_Score, Sq3_Slope, Sq4_Density
                # We have: Sq1_displacement_distance, Sq3_slope, Sq4_pixel_density

                col_name = f"{prefix}_{feature_name}"
                row_data[col_name] = value

        # Determine headers
        file_exists = os.path.isfile(output_csv)

        # We need to know all possible headers. Since this depends on dynamic keys,
        # we should ideally define a fixed schema.
        # For now, we'll collect keys from this row.
        # NOTE: If different rows have different keys (e.g. errors), CSV structure might break.
        # We will enforce a standard set of keys based on what we know, plus dynamic ones.

        # However, DictWriter can handle extras if configured, but for a consistent CSV we want all columns.
        # Let's define a standard schema based on known features.

        fieldnames = ["Filename", "Anxiety_Prediction"]
        # Sq1
        fieldnames.extend(["Sq1_centroid_x", "Sq1_centroid_y", "Sq1_displacement_distance", "Sq1_displacement_angle"])
        # Sq3
        fieldnames.extend(["Sq3_slope", "Sq3_intercept", "Sq3_r_value"])
        # Sq4
        fieldnames.extend(["Sq4_pixel_density", "Sq4_roi_density", "Sq4_heavily_shaded"])
        # Sq5
        fieldnames.extend(["Sq5_intersection_count", "Sq5_average_line_length", "Sq5_line_count"])

        # Add any other found keys to fieldnames if they aren't there (dynamic expansion is tricky with 'a' mode)
        # We will stick to the known schema to ensure stability.

        with open(output_csv, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')

            if not file_exists:
                writer.writeheader()

            writer.writerow(row_data)

    def generate_visual_report(self, original_image_path: str, analysis_results: Dict[int, Dict[str, Any]], output_dir: str = "reports/"):
        """
        Creates a visual report by annotating the original image with analysis features.

        Args:
            original_image_path: Path to the input image.
            analysis_results: Dictionary containing features.
            output_dir: Directory to save the report.
        """
        os.makedirs(output_dir, exist_ok=True)

        try:
            # re-process to get the squares (images)
            # We use the preprocessor. Note: we need to ensure we get the same processing as the analysis.
            squares = self.preprocessor.process(original_image_path)
        except Exception as e:
            print(f"Error processing image for report: {e}")
            return

        # We will reconstruct the grid or save individual images.
        # A grid is nicer. Let's create a canvas.
        # Assuming 2 rows, 4 cols.

        # Get dimensions of one square to estimate canvas size
        if not squares:
            return

        sample_sq = squares[1]
        h, w = sample_sq.shape[:2]

        # Canvas: 2 rows, 4 cols
        rows = 2
        cols = 4
        canvas_h = h * rows
        canvas_w = w * cols

        # Create color canvas (BGR)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for sq_id, image in squares.items():
            # Calculate position
            r = (sq_id - 1) // cols
            c = (sq_id - 1) % cols

            y_start = r * h
            x_start = c * w

            # Convert grayscale/binary to BGR for annotation
            if len(image.shape) == 2:
                img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                img_color = image.copy()

            # Annotate
            self._annotate_square(img_color, sq_id, analysis_results.get(sq_id, {}))

            # Resize if necessary (should match but safety check)
            img_h, img_w = img_color.shape[:2]
            if img_h != h or img_w != w:
                img_color = cv2.resize(img_color, (w, h))

            canvas[y_start:y_start+h, x_start:x_start+w] = img_color

        # Save
        filename = os.path.basename(original_image_path)
        save_path = os.path.join(output_dir, f"report_{filename}")
        cv2.imwrite(save_path, canvas)
        print(f"Saved visual report to {save_path}")

    def _annotate_square(self, image: np.ndarray, sq_id: int, result: Dict[str, Any]):
        """
        Draws annotations on a single square.
        """
        features = result.get("features", {})
        h, w = image.shape[:2]

        # 1. Draw contours (Green) - Request: "green for contours"
        # We'll just draw contours of the white ink
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Threshold to find ink (assuming white on black)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 1) # Green

        # Specific annotations
        if sq_id == 1:
            # Centroid and displacement
            cx = int(features.get("centroid_x", w/2))
            cy = int(features.get("centroid_y", h/2))
            center_x, center_y = w // 2, h // 2

            # Draw center
            cv2.circle(image, (center_x, center_y), 3, (255, 255, 0), -1) # Cyan center
            # Draw centroid
            cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1) # Red centroid
            # Draw displacement vector
            cv2.arrowedLine(image, (center_x, center_y), (cx, cy), (0, 0, 255), 2)

        elif sq_id == 3:
            # Regression line (Red)
            slope = features.get("slope", 0)
            intercept = features.get("intercept", 0)

            # y = slope * x + intercept
            # Calculate points at x=0 and x=w
            pt1_x = 0
            pt1_y = int(intercept)
            pt2_x = w
            pt2_y = int(slope * w + intercept)

            cv2.line(image, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 0, 255), 2)

        elif sq_id == 4:
            # ROI Box (Blue)
            roi = features.get("roi_coords") # [y1, y2, x1, x2]
            if roi:
                y1, y2, x1, x2 = roi
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Heavily shaded text
            if features.get("heavily_shaded"):
                cv2.putText(image, "SHADED", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        elif sq_id == 5:
            # Hough Lines (Red)
            lines = features.get("lines", [])
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
