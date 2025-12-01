import os
import pandas as pd
import numpy as np
import cv2
from scipy import stats
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from src.features.color import ColorAnalyzer
from src.features.processing import estimate_pressure, estimate_line_thickness, to_grayscale

class DataLinker:
    """
    Links image data with ground truth labels from a CSV file.
    """
    def __init__(self, image_folder: str, csv_path: str):
        self.image_folder = image_folder
        self.csv_path = csv_path
        self.linked_data = None

    def link_data(self) -> pd.DataFrame:
        """
        Matches images to CSV rows by ID.
        Assumes the 'id' column in CSV corresponds to image filenames (with or without extension).
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Clean ID column and ensure string type
        df['id'] = df['id'].astype(str).str.strip()

        image_files = os.listdir(self.image_folder)
        # Create a map of {id: full_filename}
        # We assume ID matches the filename stem or the full filename
        id_to_file = {}

        for f in image_files:
            name, ext = os.path.splitext(f)
            if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                id_to_file[name] = f
                id_to_file[f] = f # Handle case where ID includes extension

        # Find matching images
        def get_image_path(row_id):
            if row_id in id_to_file:
                return os.path.join(self.image_folder, id_to_file[row_id])
            return None

        df['image_path'] = df['id'].apply(get_image_path)

        # Drop rows where image was not found
        self.linked_data = df.dropna(subset=['image_path']).reset_index(drop=True)

        print(f"Linked {len(self.linked_data)} images out of {len(df)} rows in CSV.")
        return self.linked_data

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the linked data into Training (80%) and Validation (20%) sets.
        """
        if self.linked_data is None:
            self.link_data()

        if len(self.linked_data) == 0:
            raise ValueError("No data linked. Cannot split.")

        train_df, val_df = train_test_split(self.linked_data, test_size=test_size, random_state=random_state)
        return train_df, val_df

class PatternDiscoverer:
    """
    Implements the 'Reverse Approach': discovering visual features that correlate with specific labels.
    """
    def __init__(self, linked_data: pd.DataFrame):
        self.data = linked_data
        self.color_analyzer = ColorAnalyzer()

    def _extract_features(self, image_path: str) -> Dict[str, float]:
        """
        Extracts a feature vector for a single image.
        Combines color analysis and basic structural metrics.
        """
        image = cv2.imread(image_path)
        if image is None:
            return {}

        features = {}

        # Color Analysis
        try:
            color_stats = self.color_analyzer.analyze_palette(image)
            features.update(color_stats)
        except Exception as e:
            print(f"Error analyzing color for {image_path}: {e}")

        # Structural Analysis (Pressure, Thickness)
        try:
            # Pressure estimation (using raw image)
            features['avg_pressure'] = estimate_pressure(image)

            # Line thickness (using binary image - assumed inverted/white-on-black for distance transform)
            gray = to_grayscale(image)
            # Binarize (simple threshold)
            # We need to ensure it's white lines on black background for the thickness function
            # Assuming typical scan is black lines on white paper -> invert
            inverted = cv2.bitwise_not(gray)
            _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)

            features['avg_line_thickness'] = estimate_line_thickness(binary)

            # Add more basic features like edge density
            edges = cv2.Canny(gray, 100, 200)
            features['edge_density'] = np.mean(edges) / 255.0

        except Exception as e:
            print(f"Error analyzing structure for {image_path}: {e}")

        return features

    def find_common_features(self, target_label: str, threshold: float) -> pd.DataFrame:
        """
        Identifies statistically significant visual features for a given label/trait.

        Args:
            target_label: Column name in the CSV (e.g., 'anxiety_score').
            threshold: Value to split the data into 'Target Group' (> threshold) and 'Normal Group' (<= threshold).

        Returns:
            A pandas DataFrame reporting significant features and their stats.
        """
        if target_label not in self.data.columns:
            raise ValueError(f"Column '{target_label}' not found in data.")

        # Split into groups
        target_group = self.data[self.data[target_label] > threshold]
        normal_group = self.data[self.data[target_label] <= threshold]

        if len(target_group) < 2 or len(normal_group) < 2:
            print("Warning: Not enough data in one of the groups to perform statistical analysis.")
            return pd.DataFrame()

        print(f"Analyzing '{target_label} > {threshold}'...")
        print(f"Target Group Size: {len(target_group)}, Normal Group Size: {len(normal_group)}")

        # Extract features for both groups
        def get_group_features(df_group):
            feature_list = []
            for _, row in df_group.iterrows():
                f = self._extract_features(row['image_path'])
                if f:
                    feature_list.append(f)
            return pd.DataFrame(feature_list)

        target_features_df = get_group_features(target_group)
        normal_features_df = get_group_features(normal_group)

        if target_features_df.empty or normal_features_df.empty:
            print("Error: Could not extract features.")
            return pd.DataFrame()

        # Compare features
        results = []
        common_columns = set(target_features_df.columns).intersection(set(normal_features_df.columns))

        for col in common_columns:
            # Clean NaNs
            t_vals = target_features_df[col].dropna()
            n_vals = normal_features_df[col].dropna()

            if len(t_vals) < 2 or len(n_vals) < 2:
                continue

            # T-test
            t_stat, p_val = stats.ttest_ind(t_vals, n_vals, equal_var=False)

            mean_target = t_vals.mean()
            mean_normal = n_vals.mean()

            # We consider it significant if p-value < 0.05
            significance = "Yes" if p_val < 0.05 else "No"

            results.append({
                "feature": col,
                "mean_target_group": mean_target,
                "mean_normal_group": mean_normal,
                "p_value": p_val,
                "significant": significance
            })

        results_df = pd.DataFrame(results)
        # Sort by p-value
        if not results_df.empty:
            results_df = results_df.sort_values("p_value")

        return results_df
