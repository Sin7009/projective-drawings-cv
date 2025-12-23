# Computational Psychodiagnostics: Projective Vision Framework

## 1. Project Overview
**Goal:** To objectively automate the interpretation of projective drawing tests (specifically the **Wartegg Drawing Test / WZT**) for children aged 3-12 using Computer Vision and Machine Learning.
**Context:** Developed as part of a PhD research project (Valyavko & Knyazev), this framework bridges the gap between qualitative clinical intuition and quantitative data science. It moves from subjective interpretation to **Evidence-Based Psychometrics**.

## 2. Scientific Methodology
The system operates on two validation levels:
1.  **Expert Imitation (Supervised):** Can the CV model replicate the scores given by a human expert (e.g., "Need for Protection" in Square 8)?
2.  **Latent Discovery (Data-Driven):** By correlating visual features (line slope, pressure, density) with objective test scores (Wechsler IQ, Temml-Dorki-Amen Anxiety), we "reverse engineer" the graphical markers of psychological states.

## 3. Key Features (Implemented)
* **Adaptive Preprocessing:** Automatic form detection, perspective correction (deskewing), and adaptive grid slicing to isolate the 8 WZT squares from raw phone photos.
* **Stroke Tokenization:** A novel vectorization engine (`src/core/vectorization.py`) that decomposes drawings into atomic units ("Dots", "Lines") and filters out scanning noise.
* **Psychometric Data Loader:** A specialized module to parse clinical datasets (`.csv`), handling Cyrillic names, diagnosis codes (ONR, FFN, Stuttering), and merging them with image data.
* **Visual Memory:** A `SymbolRegistry` to store and recognize recurring semantic patterns (Few-Shot Learning ready).

## 4. Architecture
The project follows a modular Object-Oriented design:
* `src/core/`: Low-level CV logic (binarization, geometry, configuration).
* `src/features/`: Feature extraction algorithms (regression slopes, density, smoothness).
* `src/training/`: Logic for linking images to CSV ground truth and running statistical T-tests.
* `src/analysis/`: The main orchestration logic for specific tests (WZT).

## 5. Usage Guide (The Workflow)

### Phase 1: Data Preparation
1.  Place your raw psychometric tables in `data/external/`.
2.  Run the helper script to generate a scanning checklist:
    ```bash
    python scripts/generate_checklist.py
    ```
    *Output:* `scan_checklist.csv` (List of expected filenames like `Ivanov_Ivan.jpg`).

### Phase 2: Scanning
1.  Scan the children's drawings.
2.  Rename files according to the checklist (to ensure they match the database IDs).
3.  Place images in `data/raw/`.

### Phase 3: Analysis & Training
Run the main pipeline to extract features and find correlations:
```bash
# Extract global features from all images
python main.py --mode analyze --data-dir data/raw --output-dir data/processed

# Train and find patterns for specific psychological traits
python -m src.training.trainer --csv-path data/external/psychometric_data.csv \
                                --image-folder data/raw \
                                --target anxiety_score \
                                --threshold 0.5
```

*System Output:* "Significant correlation found: Heavy shading in Square 4 correlates with High Anxiety (p \< 0.05)."

### Example: Analyzing a Single Image
```python
from src.analysis.wzt_analyzer import WarteggAnalyzer
from pathlib import Path

# Initialize analyzer
analyzer = WarteggAnalyzer()

# Process a Wartegg drawing
results = analyzer.process_and_analyze("data/raw/child_001.jpg")

# Examine results for each square
for square_id, analysis in results.items():
    print(f"Square {square_id}:")
    print(f"  Tokens found: {analysis['token_count']}")
    print(f"  Features: {analysis['features']}")
    if analysis['semantic_match']:
        print(f"  Matched symbol: {analysis['semantic_match']['label']}")
```

## 6. Configuration

All thresholds (blur detection, binarization levels) are managed in `config/settings.yaml`, allowing calibration for different scanners without changing code.

### Example Configuration
```yaml
preprocessing:
  binary_threshold: 127
  blur_threshold: 100.0
  target_width: 2000

vectorization:
  dot_max_area: 100
  line_aspect_ratio: 3.0

wzt:
  grid:
    rows: 2
    cols: 4
    margins_percent: 0.1
```

## 7. Code Quality

This project follows Python best practices:
- **Type hints** for all functions
- **Comprehensive docstrings** with psychological context
- **Input validation** to prevent errors
- **Named constants** instead of magic numbers
- **Proper error handling** with specific exceptions

See [BEST_PRACTICES.md](BEST_PRACTICES.md) for coding guidelines and [IMPROVEMENTS.md](IMPROVEMENTS.md) for recent enhancements.

## 8. Testing

Run tests to verify functionality:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_feature_extraction.py

# Run with coverage
pytest --cov=src tests/
```

## 9. API Documentation

### Feature Extraction
```python
from src.features.extraction import FeatureExtractor
from pathlib import Path

# Extract global features (pressure, entropy, texture)
features = FeatureExtractor.extract_global_features(Path("image.jpg"))
print(f"Mean pressure: {features['mean_pressure']}")
print(f"Entropy: {features['entropy']}")

# Extract square-specific features
import cv2
square_image = cv2.imread("square_1.jpg", cv2.IMREAD_GRAYSCALE)
square_features = FeatureExtractor.extract_square_1_features(square_image)
print(f"Centroid displacement: {square_features['displacement_distance']}")
```

### Image Processing
```python
from src.features.processing import ImagePreprocessor

# Process Wartegg drawing
preprocessor = ImagePreprocessor()
squares = preprocessor.process("wartegg_drawing.jpg")

# Access individual squares (1-8)
for square_id, image in squares.items():
    print(f"Square {square_id} shape: {image.shape}")
```

## 10. References

Based on the research by S.M. Valyavko and K.E. Knyazev (2014), including "Possibilities of using projective methods..." and "The Forgotten but not Lost Test".

## 11. Contributing

Contributions are welcome! Please:
1. Follow the coding standards in [BEST_PRACTICES.md](BEST_PRACTICES.md)
2. Add tests for new features
3. Update documentation
4. Run linting and tests before submitting

## 12. License

See LICENSE file for details.

## 13. Acknowledgments

This framework was developed as part of a PhD research project bridging clinical psychology and computer vision to bring evidence-based methodologies to projective test interpretation.
