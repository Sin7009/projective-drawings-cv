import argparse
import os
import sys
import glob
import pandas as pd
from src.training.data_loader import PsychometricDataLoader
from src.analysis.wzt_analyzer import WarteggAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Wartegg Drawing Test Analysis Tool")
    parser.add_argument('mode', choices=['generate_list', 'analyze'], help="Operation mode")
    parser.add_argument('--data_dir', default='data/raw', help="Directory containing images (default: data/raw)")
    parser.add_argument('--output_csv', default='scan_checklist.csv', help="Output filename for checklist (default: scan_checklist.csv)")
    parser.add_argument('--norma_csv', default='Norma_Svodnaya.csv', help="Path to Norma CSV")
    parser.add_argument('--onr_csv', default='ONR_Svodnaya.csv', help="Path to ONR CSV")

    args = parser.parse_args()

    if args.mode == 'generate_list':
        print("Generating scan checklist...")
        loader = PsychometricDataLoader()

        # Check if CSVs exist
        if not os.path.exists(args.norma_csv) or not os.path.exists(args.onr_csv):
             # Try checking in data/ folder if not found in root
             norma_path = os.path.join('data', args.norma_csv)
             onr_path = os.path.join('data', args.onr_csv)

             if os.path.exists(norma_path) and os.path.exists(onr_path):
                 args.norma_csv = norma_path
                 args.onr_csv = onr_path
             else:
                 print(f"Error: Input CSVs not found. Looked for {args.norma_csv} and {args.onr_csv}")

        try:
            loader.load_and_merge(args.norma_csv, args.onr_csv)
            loader.generate_filename_checklist(output_csv=args.output_csv)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            sys.exit(1)

    elif args.mode == 'analyze':
        print(f"Analyzing images in {args.data_dir}...")
        analyzer = WarteggAnalyzer()

        # Find images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(args.data_dir, ext)))
            # Also check case-insensitive if on Linux? Glob is case sensitive.
            image_files.extend(glob.glob(os.path.join(args.data_dir, ext.upper())))

        image_files = sorted(list(set(image_files)))

        if not image_files:
            print(f"No images found in {args.data_dir}")
            sys.exit(0)

        print(f"Found {len(image_files)} images.")

        results = []
        for img_path in image_files:
            print(f"Processing {img_path}...")
            try:
                analysis_result = analyzer.process_and_analyze(img_path)
                # Flatten result for reporting/logging
                flat_res = {'filename': os.path.basename(img_path)}
                for sq_id, data in analysis_result.items():
                    flat_res[f'sq{sq_id}_mode'] = data.get('analysis_mode')
                    if 'features' in data:
                        for feat_name, feat_val in data['features'].items():
                            if isinstance(feat_val, (int, float, str, bool)):
                                flat_res[f'sq{sq_id}_{feat_name}'] = feat_val
                results.append(flat_res)
            except Exception as e:
                print(f"Failed to analyze {img_path}: {e}")

        if results:
            df_results = pd.DataFrame(results)
            output_file = "analysis_results.csv"
            df_results.to_csv(output_file, index=False)
            print(f"Analysis complete. Results saved to {output_file}")
        else:
            print("No results generated.")

if __name__ == "__main__":
    main()
