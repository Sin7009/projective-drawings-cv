import pytest
import pandas as pd
import os
from src.training.data_loader import PsychometricDataLoader

@pytest.fixture
def dummy_data(tmp_path):
    norma_csv = tmp_path / "norma.csv"
    onr_csv = tmp_path / "onr.csv"

    df_norma = pd.DataFrame({
        'Ф.И.О': ['Иван', 'Петр'],
        'Тревожность Дорки, Амен, Теммл': [10, 20]
    })
    df_onr = pd.DataFrame({
        'Ф.И.О': ['Сидор'],
        'Тревожность Дорки, Амен, Теммл': [30]
    })

    df_norma.to_csv(norma_csv, index=False)
    df_onr.to_csv(onr_csv, index=False)

    return str(norma_csv), str(onr_csv)

def test_generate_filename_checklist(dummy_data, tmp_path):
    norma_path, onr_path = dummy_data
    loader = PsychometricDataLoader()
    loader.load_and_merge(norma_path, onr_path)

    output_csv = tmp_path / "checklist.csv"
    checklist = loader.generate_filename_checklist(output_csv=str(output_csv))

    assert isinstance(checklist, pd.DataFrame)
    assert len(checklist) == 3
    assert 'Original' in checklist.columns
    assert 'Expected_Filename' in checklist.columns

    # Check transliteration
    # 'Иван' -> 'Ivan.jpg'
    # 'Петр' -> 'Petr.jpg'
    # 'Сидор' -> 'Sidor.jpg'

    # Note: Actual transliteration depends on the map in PsychometricDataLoader.
    # 'И' -> 'I', 'в' -> 'v', 'а' -> 'a', 'н' -> 'n' => 'Ivan'
    assert checklist[checklist['Original'] == 'Иван']['Expected_Filename'].iloc[0] == 'Ivan.jpg'

    assert os.path.exists(output_csv)

def test_generate_filename_checklist_no_args(dummy_data):
    norma_path, onr_path = dummy_data
    loader = PsychometricDataLoader()
    loader.load_and_merge(norma_path, onr_path)

    checklist = loader.generate_filename_checklist()

    assert isinstance(checklist, pd.DataFrame)
    assert len(checklist) == 3

def test_generate_filename_checklist_not_loaded():
    loader = PsychometricDataLoader()
    with pytest.raises(ValueError, match="Dataframe not loaded"):
        loader.generate_filename_checklist()
