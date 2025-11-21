import os
import shutil
import pandas as pd
import numpy as np
import cv2
import pytest
from src.training.trainer import DataLinker, PatternDiscoverer

# Setup temporary directory for testing
TEST_DIR = "temp_test_data"
IMG_DIR = os.path.join(TEST_DIR, "images")
CSV_PATH = os.path.join(TEST_DIR, "ground_truth.csv")

def setup_module(module):
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(IMG_DIR)

    # Create dummy images
    # 1. High Anxiety (Dark, Thick lines)
    for i in range(5):
        img = np.full((100, 100, 3), 255, dtype=np.uint8) # White background
        # Draw thick black lines
        cv2.line(img, (10, 10), (90, 90), (0, 0, 0), 5)
        cv2.line(img, (90, 10), (10, 90), (0, 0, 0), 5)
        cv2.imwrite(os.path.join(IMG_DIR, f"scan_{i:03d}.jpg"), img)

    # 2. Low Anxiety (Light, Thin lines)
    for i in range(5, 10):
        img = np.full((100, 100, 3), 255, dtype=np.uint8) # White background
        # Draw thin gray lines (lighter pressure)
        cv2.line(img, (10, 10), (90, 90), (100, 100, 100), 1)
        cv2.line(img, (90, 10), (10, 90), (100, 100, 100), 1)
        cv2.imwrite(os.path.join(IMG_DIR, f"scan_{i:03d}.jpg"), img)

    # Create CSV
    data = {
        'id': [f"scan_{i:03d}" for i in range(10)],
        'anxiety_score': [80, 85, 90, 82, 88, 20, 30, 25, 35, 40], # First 5 high, last 5 low
        'iq_score': [100] * 10
    }
    df = pd.DataFrame(data)
    df.to_csv(CSV_PATH, index=False)

def teardown_module(module):
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

def test_datalinker():
    linker = DataLinker(IMG_DIR, CSV_PATH)
    df = linker.link_data()

    assert len(df) == 10
    assert 'image_path' in df.columns
    assert os.path.exists(df.iloc[0]['image_path'])

    train, val = linker.split_data(test_size=0.2)
    assert len(train) == 8
    assert len(val) == 2

def test_pattern_discoverer():
    linker = DataLinker(IMG_DIR, CSV_PATH)
    linker.link_data()

    discoverer = PatternDiscoverer(linker.linked_data)

    # Check high anxiety (threshold 50)
    # We expect significant difference in pressure or thickness
    report = discoverer.find_common_features(target_label="anxiety_score", threshold=50)

    print("\nReport:\n", report)

    assert not report.empty
    assert 'feature' in report.columns
    assert 'p_value' in report.columns

    # Check if we found significant features
    significant = report[report['significant'] == 'Yes']
    assert not significant.empty

    # Pressure or thickness should be significant
    feats = significant['feature'].tolist()
    assert 'avg_pressure' in feats or 'avg_line_thickness' in feats

if __name__ == "__main__":
    # Manually run setup and tests if run as script
    setup_module(None)
    try:
        test_datalinker()
        print("DataLinker Test Passed")
        test_pattern_discoverer()
        print("PatternDiscoverer Test Passed")
    except Exception as e:
        print(f"Tests Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        teardown_module(None)
