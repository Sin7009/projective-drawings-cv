import pandas as pd
import os
from scipy import stats
import numpy as np
from typing import Optional, List, Dict

class PsychometricDataLoader:
    """
    Loads and processes psychometric data from CSV files, and links them with image data.
    """

    # Speech Status (Реч_статус)
    DIAGNOSIS_MAP = {
        1: "ONR",        # General Speech Underdevelopment
        2: "Norm",       # Normal
        3: "FFN",        # Phonetic-Phonemic
        4: "Stuttering"  # Logoneurosis
    }

    # Gender (Пол)
    GENDER_MAP = {
        0: "Female",
        1: "Male"
    }

    # Column mapping from Russian CSV headers to internal English variable names
    COLUMN_MAPPING = {
        "Тревожность Дорки, Амен, Теммл": "target_anxiety",
        "Таблицы Агрессивность": "target_aggression",
        "Таблицы Контакт": "target_social_contact",
        "Возраст": "meta_age",
        "Пол (0 - Ж; 1 - М)": "meta_gender",
        "Реч_статус": "target_diagnosis"
    }

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None

    def load_and_merge(self, path_norma: str, path_onr: str, fill_na_strategy: str = 'mean') -> pd.DataFrame:
        """
        Reads both CSVs, standardizes columns, and merges them.

        Args:
            path_norma: Path to the Norma_Svodnaya.csv file.
            path_onr: Path to the ONR_Svodnaya.csv file.
            fill_na_strategy: Strategy to handle missing values ('mean', 'drop', or None).
                              'mean' fills NaNs in numeric columns with the column mean,
                              EXCEPT for target columns which are preserved.

        Returns:
            pd.DataFrame: The merged and standardized dataframe.
        """
        df_norma = pd.read_csv(path_norma)
        df_onr = pd.read_csv(path_onr)

        # Add a source column to distinguish datasets if needed
        df_norma['dataset_source'] = 'norma'
        df_onr['dataset_source'] = 'onr'

        # Merge datasets
        combined_df = pd.concat([df_norma, df_onr], ignore_index=True)

        # Rename columns
        combined_df.rename(columns=self.COLUMN_MAPPING, inplace=True)

        # Apply Value Mappings
        if 'target_diagnosis' in combined_df.columns:
            combined_df['target_diagnosis'] = combined_df['target_diagnosis'].map(self.DIAGNOSIS_MAP)

        if 'meta_gender' in combined_df.columns:
            combined_df['meta_gender'] = combined_df['meta_gender'].map(self.GENDER_MAP)

        # Handle Missing Values
        target_columns = set(self.COLUMN_MAPPING.values())

        if fill_na_strategy == 'drop':
            combined_df.dropna(subset=[col for col in target_columns if col in combined_df.columns], inplace=True)
        elif fill_na_strategy == 'mean':
            for col in combined_df.columns:
                # Do NOT fill NaNs in target columns
                if col in target_columns:
                    continue

                # Only fill numeric columns with mean
                if pd.api.types.is_numeric_dtype(combined_df[col]):
                    combined_df[col] = combined_df[col].fillna(combined_df[col].mean())

        self.df = combined_df
        return combined_df

    def match_images(self, dataframe: Optional[pd.DataFrame] = None, image_dir: str = "") -> pd.DataFrame:
        """
        Matches images in the directory to the dataframe based on 'Ф.И.О' column.
        Adds 'image_path' column to the dataframe.

        Args:
            dataframe: The dataframe containing psychometric data. If None, uses self.df.
            image_dir: Directory containing the images.

        Returns:
            pd.DataFrame: Dataframe with 'image_path' column.
        """
        if dataframe is None:
            if self.df is None:
                raise ValueError("Dataframe not loaded. Call load_and_merge first or provide a dataframe.")
            dataframe = self.df
        else:
            # If a dataframe is provided, we work on it.
            # Should we update self.df? Let's assume yes if it matches self.df in identity or content,
            # but to be safe we just return the modified one.
            pass

        if 'Ф.И.О' not in dataframe.columns:
            raise ValueError("Column 'Ф.И.О' not found in dataframe.")

        def find_image(name):
            if not isinstance(name, str):
                return None

            # Transliterate name to match filename format
            transliterated_name = self._transliterate(name)

            # Check for common extensions
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                filename = f"{transliterated_name}{ext}"
                filepath = os.path.join(image_dir, filename)
                if os.path.exists(filepath):
                    return filepath
            return None

        dataframe['image_path'] = dataframe['Ф.И.О'].apply(find_image)

        # Update self.df if we are working on the internal state
        if self.df is not None and dataframe is self.df:
            self.df = dataframe
        elif self.df is not None and dataframe.equals(self.df): # This might be expensive
            self.df = dataframe

        # If the user passed a dataframe that is likely the one we just loaded, update self.df
        # To be safe, let's just update self.df if it was None, or if the user wants to rely on state.
        # The prompt implies we should update the state.
        # Let's just update self.df to the result if self.df was set.
        if self.df is not None:
             # We might be replacing it with a version that has image_path
             self.df = dataframe

        return dataframe

    def get_clean_dataset(self, target_col: str) -> pd.DataFrame:
        """
        Returns only the rows where target_col is not NaN and image_path exists.
        This ensures we train on valid ground truth only.
        """
        if self.df is None:
             raise ValueError("Dataframe not loaded. Call load_and_merge first.")

        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")

        if 'image_path' not in self.df.columns:
            raise ValueError("Column 'image_path' not found. Call match_images first.")

        # Filter: target not NaN AND image_path not None
        clean_df = self.df[self.df[target_col].notna() & self.df['image_path'].notna()]

        return clean_df

    def generate_expected_filenames(self, output_csv: str = 'filenames.csv'):
        """
        Iterates through the dataframe and generates a list of expected filenames based on the 'Ф.И.О' column.
        """
        if self.df is None:
             raise ValueError("Dataframe not loaded. Call load_and_merge first.")

        if 'Ф.И.О' not in self.df.columns:
            raise ValueError("Column 'Ф.И.О' not found in dataframe.")

        expected_files = []
        for name in self.df['Ф.И.О']:
            if isinstance(name, str):
                t_name = self._transliterate(name)
                expected_files.append({'Original': name, 'Expected_Filename': t_name + ".jpg"}) # Assuming .jpg default
            else:
                expected_files.append({'Original': name, 'Expected_Filename': None})

        out_df = pd.DataFrame(expected_files)
        out_df.to_csv(output_csv, index=False)
        print(f"Expected filenames saved to {output_csv}")

    def generate_filename_checklist(self, output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Generates a checklist of filenames that should be used when scanning, based on the subjects' names.
        This helps in renaming the scanned files to match what the loader expects.

        Args:
            output_csv: Optional path to save the checklist.

        Returns:
            pd.DataFrame: A dataframe containing 'Original' name and 'Expected_Filename'.
        """
        if self.df is None:
             raise ValueError("Dataframe not loaded. Call load_and_merge first.")

        if 'Ф.И.О' not in self.df.columns:
            raise ValueError("Column 'Ф.И.О' not found in dataframe.")

        checklist_data = []
        for name in self.df['Ф.И.О']:
            if isinstance(name, str):
                t_name = self._transliterate(name)
                checklist_data.append({'Original': name, 'Expected_Filename': t_name + ".jpg"})
            else:
                checklist_data.append({'Original': name, 'Expected_Filename': None})

        checklist_df = pd.DataFrame(checklist_data)

        if output_csv:
            checklist_df.to_csv(output_csv, index=False)
            print(f"Filename checklist saved to {output_csv}")

        return checklist_df

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
        Spaces are replaced with underscores.
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
            'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
            ' ': '_' # Replace space with underscore
        }

        result = []
        for char in text:
            result.append(ru_en_map.get(char, char))
        return "".join(result)
