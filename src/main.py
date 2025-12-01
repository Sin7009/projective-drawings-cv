import typer
from pathlib import Path
from loguru import logger
import pandas as pd
from typing import Optional
from src.features.extraction import FeatureExtractor
import sys

app = typer.Typer()

@app.command()
def analyze(
    data_dir: Path = typer.Option(Path("data/raw"), help="Directory containing input images"),
    output_dir: Path = typer.Option(Path("data/processed"), help="Directory for output CSV"),
    output_filename: str = typer.Option("features.csv", help="Output CSV filename")
):
    """
    Extracts global features from all images in the data directory and saves to CSV.
    """
    logger.info(f"Starting analysis on {data_dir}...")

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename

    results = []

    # Supported extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in extensions:
        # Recursive glob if needed, but standard is flat here
        image_files.extend(list(data_dir.glob(ext)))
        image_files.extend(list(data_dir.glob(ext.upper())))

    # Remove duplicates
    image_files = sorted(list(set(image_files)))

    if not image_files:
        logger.warning(f"No images found in {data_dir}")
        raise typer.Exit(code=0)

    logger.info(f"Found {len(image_files)} images. Processing...")

    for img_path in image_files:
        logger.debug(f"Processing {img_path.name}...")
        try:
            feats = FeatureExtractor.extract_global_features(img_path)
            if feats:
                results.append(feats)
        except Exception as e:
            logger.error(f"Unexpected error processing {img_path}: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        logger.success(f"Analysis complete. Processed {len(df)} files. Results saved to {output_file}")
        print(df.head())
    else:
        logger.warning("No features extracted.")

if __name__ == "__main__":
    app()
