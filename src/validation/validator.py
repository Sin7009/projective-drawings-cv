from typing import Dict, List, Union
import numpy as np
from scipy.stats import pearsonr

class ConvergentValidator:
    """
    Implements convergent validation strategies to correlate Computer Vision metrics
    with standard psychometric data.
    Ref: Valyavko & Knyazev (2014).
    """

    def __init__(self):
        pass

    def correlate(self, cv_scores: List[float], external_test_scores: List[float]) -> float:
        """
        Calculates the Pearson correlation coefficient between CV-derived scores
        and external psychometric test scores.

        Args:
            cv_scores: List of scores derived from the computer vision analysis.
            external_test_scores: List of scores from standard psychometric tests (e.g., Anxiety scale).

        Returns:
            The Pearson correlation coefficient (r).
            Returns 0.0 if lists are empty or have different lengths (logs warning).
        """
        if not cv_scores or not external_test_scores:
            print("Warning: Empty score lists provided to correlate.")
            return 0.0

        if len(cv_scores) != len(external_test_scores):
            print(f"Warning: Score lists have different lengths ({len(cv_scores)} vs {len(external_test_scores)}). Cannot calculate correlation.")
            return 0.0

        # Ensure inputs are numpy arrays
        cv_arr = np.array(cv_scores)
        ext_arr = np.array(external_test_scores)

        # Check for constant input (variance is 0), which causes Pearsonr to raise an error or return nan
        if np.std(cv_arr) == 0 or np.std(ext_arr) == 0:
            print("Warning: One of the input arrays is constant. Correlation is undefined.")
            return 0.0

        try:
            correlation, _ = pearsonr(cv_arr, ext_arr)
            if np.isnan(correlation):
                return 0.0
            return float(correlation)
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            return 0.0
