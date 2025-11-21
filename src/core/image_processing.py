import cv2
import numpy as np
from typing import Tuple, List, Dict

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def binarize(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Apply binary thresholding."""
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def estimate_line_thickness(binary_image: np.ndarray) -> float:
    """
    Estimate average line thickness using distance transform.
    Assumes white lines on black background or inverted binary.
    """
    # Ensure we are working with white strokes on black background
    if np.mean(binary_image) > 127:
        # Input is likely black text on white bg, invert it
        binary_image = cv2.bitwise_not(binary_image)

    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    # Skeletonize to find center of lines
    skeleton = cv2.ximgproc.thinning(binary_image)

    # Get thickness values along the skeleton
    thickness_values = dist_transform[skeleton > 0] * 2 # radius * 2

    if len(thickness_values) == 0:
        return 0.0

    return float(np.mean(thickness_values))

def estimate_pressure(image: np.ndarray) -> float:
    """
    Estimate pen pressure based on pixel intensity in grayscale image.
    Lower intensity values (darker pixels) imply higher pressure.
    Returns a normalized pressure value (0-1).
    """
    gray = to_grayscale(image)

    # Invert so that dark pixels (ink) are high values
    inverted = 255 - gray

    # Mask out background (assume background is mostly 0 in inverted)
    # Simple threshold to identify 'ink'
    mask = inverted > 20

    if not np.any(mask):
        return 0.0

    ink_pixels = inverted[mask]
    avg_intensity = np.mean(ink_pixels)

    return float(avg_intensity / 255.0)

class ImagePreprocessor:
    """
    Handles the raw image intake: quality check, perspective correction,
    and extraction of the 8 Wartegg squares.
    """

    def __init__(self, target_width=2000, target_height=1414):
        # A4 aspect ratio approx sqrt(2) -> 1.414
        self.width = target_width
        self.height = target_height

    def check_blur(self, image: np.ndarray, threshold: float = 100.0) -> bool:
        """
        Returns True if the image is too blurry using Laplacian Variance.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return score < threshold

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Orders coordinates: top-left, top-right, bottom-right, bottom-left.
        Necessary for perspective transform.
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)] # Bottom-right has largest sum

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right has smallest diff
        rect[3] = pts[np.argmax(diff)] # Bottom-left has largest diff
        return rect

    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Applies perspective transformation to obtain a top-down view.
        """
        rect = self.order_points(pts)

        # Construct standard destination points
        dst = np.array([
            [0, 0],
            [self.width - 1, 0],
            [self.width - 1, self.height - 1],
            [0, self.height - 1]], dtype="float32")

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (self.width, self.height))
        return warped

    def find_paper_contour(self, image: np.ndarray) -> np.ndarray:
        """
        Finds the largest 4-sided contour assumed to be the paper sheet.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            # Approx the contour to a polygon
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # If it has 4 points, we assume it's our paper
            if len(approx) == 4:
                return approx.reshape(4, 2)

        raise ValueError("Could not find the document contour. Ensure contrast with background.")

    def slice_grid(self, warped_image: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Slices the rectified image into 8 squares based on Wartegg grid layout.
        Assumes standard margins and gaps (needs tuning based on your specific blank).
        """
        # Hardcoded percentages for a standard WZT layout.
        # These need to be calibrated once against your specific template.
        # Example logic:
        rows = 2
        cols = 4

        # Dimensions of one cell (approx)
        cell_w = self.width // cols
        cell_h = self.height // 3 # Assuming the grid takes up top 2/3rds
        # Note: The user provided code uses //3. This implies the bottom 1/3 is header/footer or empty.
        # I will stick to their code.

        squares = {}
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c + 1

                # Add margins to crop out the black borders
                margin_x = int(cell_w * 0.1)
                margin_y = int(cell_h * 0.1)

                x1 = c * cell_w + margin_x
                y1 = r * cell_h + margin_y
                x2 = (c + 1) * cell_w - margin_x
                y2 = (r + 1) * cell_h - margin_y

                roi = warped_image[y1:y2, x1:x2]
                squares[idx] = roi

        return squares

    def process(self, image_path: str) -> Dict[int, np.ndarray]:
        """Main pipeline."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        if self.check_blur(image):
            print("Warning: Image might be blurry.")

        contour = self.find_paper_contour(image)
        warped = self.four_point_transform(image, contour)
        # Binarize after warping for cleaner lines
        warped_bin = binarize(to_grayscale(warped))

        return self.slice_grid(warped_bin)
