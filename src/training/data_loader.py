import pandas as pd
import os
from scipy import stats
import numpy as np
from typing import Optional, List, Dict

class PsychometricDataLoader:
    """
    Loads and processes psychometric data from CSV files, and links them with image data.
    """

    # Column mapping from Russian CSV headers to internal English variable names
    COLUMN_MAPPING = {
        "Тревожность Дорки, Амен, Теммл": "target_anxiety",
        "Таблицы Агрессивность": "target_aggression",
        "Таблицы Контакт": "target_social_contact",
        "Возраст": "meta_age",
        "Пол (0 - Ж; 1 - М)": "meta_gender"
    }

    def __init__(self):
        pass

    def load_and_merge(self, path_norma: str, path_onr: str, fill_na_strategy: str = 'mean') -> pd.DataFrame:
        """
        Reads both CSVs, standardizes columns, and merges them.

        Args:
            path_norma: Path to the Norma_Svodnaya.csv file.
            path_onr: Path to the ONR_Svodnaya.csv file.
            fill_na_strategy: Strategy to handle missing values ('mean', 'drop', or None).
                              'mean' fills NaNs in numeric columns with the column mean.
                              'drop' drops rows with NaNs.

        Returns:
            pd.DataFrame: The merged and standardized dataframe.
        """
        df_norma = pd.read_csv(path_norma)
        df_onr = pd.read_csv(path_onr)

        # Add a source column to distinguish datasets if needed, or just merge
        df_norma['dataset_source'] = 'norma'
        df_onr['dataset_source'] = 'onr'

        # Merge datasets
        combined_df = pd.concat([df_norma, df_onr], ignore_index=True)

        # Rename columns
        combined_df.rename(columns=self.COLUMN_MAPPING, inplace=True)

        # Handle Missing Values
        if fill_na_strategy == 'drop':
            combined_df.dropna(subset=self.COLUMN_MAPPING.values(), inplace=True)
        elif fill_na_strategy == 'mean':
            for col in self.COLUMN_MAPPING.values():
                if col in combined_df.columns:
                    # Only fill numeric columns with mean
                    if pd.api.types.is_numeric_dtype(combined_df[col]):
                        combined_df[col] = combined_df[col].fillna(combined_df[col].mean())

        return combined_df

    def match_images(self, dataframe: pd.DataFrame, image_dir: str) -> pd.DataFrame:
        """
        Matches images in the directory to the dataframe based on 'Ф.И.О' column.
        Adds 'image_path' column to the dataframe.

        Args:
            dataframe: The dataframe containing psychometric data.
            image_dir: Directory containing the images.

        Returns:
            pd.DataFrame: Dataframe with 'image_path' column.
        """
        if 'Ф.И.О' not in dataframe.columns:
            raise ValueError("Column 'Ф.И.О' not found in dataframe.")

        def find_image(name):
            if not isinstance(name, str):
                return None

            # Transliterate name to match filename format
            # This is a simplified transliteration, assuming standard mapping.
            transliterated_name = self._transliterate(name)

            # Try to find the file (assuming jpg, png, etc.)
            # We search for files that start with the transliterated name
            # Or we can try exact match if extensions are known.
            # Let's try a flexible search in the directory.

            # Simple approach: check for common extensions
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                filename = f"{transliterated_name}{ext}"
                filepath = os.path.join(image_dir, filename)
                if os.path.exists(filepath):
                    return filepath

            # Alternative: check if transliterated name is a substring of any file in dir?
            # The requirement says: "Assume image filenames correspond to the `Ф.И.О` column (transliterated)"
            # So exact match of name part is likely intended.

            return None

        dataframe['image_path'] = dataframe['Ф.И.О'].apply(find_image)
        return dataframe

    def analyze_correlations(self, dataframe: pd.DataFrame, target_column: str = 'target_anxiety',
                             feature_columns: Optional[List[str]] = None) -> Dict:
        """
        Splits the data into high/low groups and performs T-test on visual features.

        Args:
            dataframe: The dataframe with data.
            target_column: The psychometric variable to split by (e.g., 'target_anxiety').
            feature_columns: List of visual feature columns to analyze.
                             If None, selects all numeric columns except targets/meta.

        Returns:
            Dict: Dictionary containing T-test results.
        """
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        # Drop rows where target is NaN
        df_clean = dataframe.dropna(subset=[target_column])

        if df_clean.empty:
            return {"error": "No data available for analysis after dropping NaNs."}

        # Determine feature columns if not provided
        if feature_columns is None:
            # Exclude known non-feature columns
            exclude_cols = set(self.COLUMN_MAPPING.values()) | {'Ф.И.О', 'image_path', 'dataset_source'}
            feature_columns = [c for c in df_clean.select_dtypes(include=[np.number]).columns
                               if c not in exclude_cols]

        if not feature_columns:
             return {"error": "No feature columns found for analysis."}

        # Split into High (Top 25%) and Low (Bottom 25%) groups
        low_threshold = df_clean[target_column].quantile(0.25)
        high_threshold = df_clean[target_column].quantile(0.75)

        low_group = df_clean[df_clean[target_column] <= low_threshold]
        high_group = df_clean[df_clean[target_column] >= high_threshold]

        results = {}

        for feature in feature_columns:
            if feature not in df_clean.columns:
                continue

            val_low = low_group[feature].dropna()
            val_high = high_group[feature].dropna()

            if len(val_low) < 2 or len(val_high) < 2:
                results[feature] = {"p_value": None, "statistic": None, "note": "Insufficient data"}
                continue

            # T-test
            t_stat, p_val = stats.ttest_ind(val_high, val_low, equal_var=False)

            results[feature] = {
                "statistic": t_stat,
                "p_value": p_val,
                "significant": p_val < 0.05 if p_val is not None else False,
                "mean_high": val_high.mean(),
                "mean_low": val_low.mean()
            }

        return results

    def _transliterate(self, text: str) -> str:
        """
        Simple transliteration from Russian to English.
        """
        ru_en_map = {
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
            'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
            'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
            'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
            'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
            'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
            'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
            'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
            'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Shch',
            'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya'
        }

        result = []
        for char in text:
            result.append(ru_en_map.get(char, char))
        return "".join(result)
