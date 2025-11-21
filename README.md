# Universal Projective Test Analysis Framework

## Abstract
This project, "Scalable Computational Psychodiagnostics," aims to develop a comprehensive framework for objectifying and automating the scoring of various projective drawing tests. By leveraging a modular Object-Oriented architecture and advanced Computer Vision (CV) techniques, this platform supports a battery of psychological tests, including the Wartegg Drawing Test (WZT), House-Tree-Person (HTP), Draw-a-Person, and more. The goal is to translate qualitative clinical observations into quantitative, reproducible metrics for research and diagnostic support.

## Methodology
The framework employs a "Universal Projective Test Analysis" approach, standardizing the pipeline across different instruments.

### Supported Tests
1.  **Wartegg Drawing Test (WZT):** Analyzes response to semi-structured graphical stimuli (integration, line quality, thematic content).
2.  **House-Tree-Person (HTP):** Detects and segments house, tree, and person entities to analyze structural attributes (size, placement, ratio).
3.  **Human Figure Drawings (Machover/DAP):** Utilizes pose estimation to analyze body proportions and specific features.
4.  **Stars and Waves / Non-existent Animal:** (Future implementation) Analysis of symbolic content and creative expression.

### Architecture
The system is built on a modular design:
-   **Core:** Shared Computer Vision utilities for preprocessing, stroke analysis (pressure, thickness), and geometric feature extraction.
-   **Test Modules:** Specialized classes for each test type inheriting from a common `ProjectiveTest` interface.
-   **AI Integration:**
    -   **MediaPipe:** For skeleton and pose estimation in human figure drawings.
    -   **Ultralytics (YOLO):** For object detection and segmentation in HTP tests.

### Advanced Diagnostic Layers
The framework now includes multimodal analysis and validation strategies:

*   **Multimodal Analysis:**
    *   **Color Analysis:** Quantifies "Warm vs. Cold" and "Dark vs. Light" ratios and detects dominant color palettes to interpret emotional tone.
    *   **Semantic Analysis:** Integrates OCR (via Tesseract) to read and interpret titles, labels, or text embedded in drawings.
*   **Validation Strategies:**
    *   **Convergent Validity:** Includes tools to correlate computer vision-derived scores with standard psychometric scales (e.g., anxiety, aggression) using Pearson correlation, following the methodology of Valyavko & Knyazev (2014).
*   **Input Flexibility:** Supports both static scans and digital tablet input (time-series data) for trajectory analysis.

### Semi-Supervised Annotation Workflow (Human-in-the-Loop)
To facilitate the creation of a structured Visual Knowledge Graph, the project includes a Streamlit-based labeling tool.

*   **Web Interface (`src/labeling/app.py`):**
    *   **Cluster View:** Automatically groups unlabeled drawings by visual similarity using CNN feature vectors (ResNet18) and K-Means clustering. This allows researchers to label coherent batches of drawings efficiently.
    *   **Input Form:** Provides fields to capture semantic data ("Child's Title") and expert diagnostic labels.
    *   **Decomposition Mode:** Enables granular annotation by allowing experts to draw bounding boxes around specific sub-parts of a drawing (e.g., "sharp teeth", "large hands"), linking visual regions to specific concepts in the ontology.
*   **Ontology (`src/core/ontology.py`):** Defines hierarchical `ConceptNode` structures to organize labels and attributes systematically.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast and efficient dependency management.

### Prerequisites
-   Python 3.9+
-   `uv` installed (see [uv documentation](https://docs.astral.sh/uv/getting-started/installation/))

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/projective-drawings-cv.git
    cd projective-drawings-cv
    ```

2.  Install dependencies:
    ```bash
    uv sync
    ```
    This will create a virtual environment and install all required packages, including `mediapipe`, `ultralytics`, `opencv`, `torch`, etc.

3.  Activate the environment:
    ```bash
    source .venv/bin/activate
    # On Windows: .venv\Scripts\activate
    ```

## Usage Structure

-   `src/core`: Shared utilities (image processing, math).
-   `src/tests`: Test implementations.
    -   `base_test.py`: Abstract Base Class definition.
    -   `wzt/`: Wartegg Drawing Test module.
    -   `htp/`: House-Tree-Person module.
    -   `human_figure/`: Human Figure module.

## Roadmap
- [ ] Implement specific feature extraction for WZT.
- [ ] Train/Integrate YOLO models for HTP object detection.
- [ ] Integrate MediaPipe for Human Figure analysis.
- [ ] Develop scoring rubrics based on standard psychological literature.
