import os
import pandas as pd
import pytest
from src.training.data_loader import PsychometricDataLoader

@pytest.fixture
def checklist_test_data(tmp_path):
    """Creates dummy CSV files for testing checklist generation."""
    norma_path = tmp_path / "Norma_Svodnaya.csv"
    onr_path = tmp_path / "ONR_Svodnaya.csv"

    norma_data = {
        'Ф.И.О': ['Иванов Иван', 'Петров Петр'],
        'Тревожность Дорки, Амен, Теммл': [1, 2],
        'Таблицы Агрессивность': [1, 1],
        'Таблицы Контакт': [5, 5],
        'Возраст': [10, 11],
        'Пол (0 - Ж; 1 - М)': [1, 1],
        'Реч_статус': [2, 2] # Norm
    }

    onr_data = {
        'Ф.И.О': ['Сидорова Анна', 'Козлов Константин'],
        'Тревожность Дорки, Амен, Теммл': [3, 4],
        'Таблицы Агрессивность': [2, 3],
        'Таблицы Контакт': [2, 1],
        'Возраст': [10, 12],
        'Пол (0 - Ж; 1 - М)': [0, 1],
        'Реч_статус': [1, 1] # ONR
    }

    pd.DataFrame(norma_data).to_csv(norma_path, index=False)
    pd.DataFrame(onr_data).to_csv(onr_path, index=False)

    return str(norma_path), str(onr_path)

def test_generate_filename_checklist(checklist_test_data, tmp_path):
    norma_path, onr_path = checklist_test_data
    output_csv = tmp_path / "scan_checklist.csv"

    loader = PsychometricDataLoader()
    loader.load_and_merge(norma_path, onr_path)

    loader.generate_filename_checklist(output_csv=str(output_csv))

    assert output_csv.exists()

    df = pd.read_csv(output_csv)

    # Verify columns
    expected_columns = ['Original_Name', 'Expected_Filename', 'Diagnosis']
    assert all(col in df.columns for col in expected_columns)

    # Verify content
    # Ivanov Ivan -> Ivanov_Ivan.jpg
    row = df[df['Original_Name'] == 'Иванов Иван'].iloc[0]
    assert row['Expected_Filename'] == 'Ivanov_Ivan.jpg'
    assert row['Diagnosis'] == 'Norm'

    # Kozlov Konstantin -> Kozlov_Konstantin.jpg
    row = df[df['Original_Name'] == 'Козлов Константин'].iloc[0]
    assert row['Expected_Filename'] == 'Kozlov_Konstantin.jpg'
    assert row['Diagnosis'] == 'ONR'
