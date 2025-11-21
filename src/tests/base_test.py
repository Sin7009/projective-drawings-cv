from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

class ProjectiveTest(ABC):
    """
    Abstract Base Class for Projective Drawing Tests.
    Defines the standard interface for loading, processing, and scoring
    different projective tests (WZT, HTP, etc.).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the test with optional configuration.

        Args:
            config: A dictionary containing test-specific parameters.
        """
        self.config = config or {}
        self.input_type = self.config.get("input_type", "scan")  # 'scan' or 'tablet'
        self.image = None
        self.preprocessed_image = None
        self.features = {}

    def trajectory_analysis(self, data: Any) -> Dict[str, Any]:
        """
        Analyze trajectory data (start/end points, pressure, velocity).
        Only active if input_type == 'tablet'.

        Args:
            data: Time-series data from tablet input.

        Returns:
            Dictionary of trajectory features.
        """
        if self.input_type != "tablet":
            return {}
        # Placeholder for actual trajectory analysis implementation
        return {"status": "Not implemented", "points_count": len(data) if hasattr(data, '__len__') else 0}

    @abstractmethod
    def load_image(self, filepath: str) -> None:
        """
        Load an image from the specified path.

        Args:
            filepath: Path to the image file.
        """
        pass

    @abstractmethod
    def preprocess(self) -> np.ndarray:
        """
        Apply standard preprocessing steps (grayscale, denoising, binarization, etc.).

        Returns:
            The preprocessed image.
        """
        pass

    @abstractmethod
    def extract_features(self) -> Dict[str, Any]:
        """
        Extract features relevant to the specific test (e.g., strokes, objects, placement).

        Returns:
            A dictionary of extracted features.
        """
        pass

    @abstractmethod
    def score(self) -> Dict[str, float]:
        """
        Calculate psychological scores based on extracted features.

        Returns:
            A dictionary of scores mapping psychological traits to quantitative values.
        """
        pass

    def run_pipeline(self, filepath: str) -> Dict[str, Any]:
        """
        Execute the full analysis pipeline: Load -> Preprocess -> Extract -> Score.

        Args:
            filepath: Path to the image file.

        Returns:
            A dictionary containing both features and scores.
        """
        self.load_image(filepath)
        self.preprocess()
        features = self.extract_features()
        scores = self.score()
        return {"features": features, "scores": scores}
