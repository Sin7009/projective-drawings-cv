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
python main.py --mode train --target anxiety
```

*System Output:* "Significant correlation found: Heavy shading in Square 4 correlates with High Anxiety (p \< 0.05)."

## 6. Configuration

All thresholds (blur detection, binarization levels) are managed in `config/settings.yaml`, allowing calibration for different scanners without changing code.

## 7. References

Based on the research by S.M. Valyavko and K.E. Knyazev (2014), including "Possibilities of using projective methods..." and "The Forgotten but not Lost Test".
