"""
main.py

Comprehensive image similarity analysis tool with batch processing capabilities.
Analyzes similarity between original and processed images using multiple computer vision
and AI techniques: pixel differences, semantic embeddings, and pose detection.

Features:
- Single pair analysis
- Batch processing from CSV files
- Directory scanning for image pairs
- Weighted scoring system optimized for body-swap analysis
- CSV output with detailed statistics

Usage:
    # Single pair
    python main.py -i input.png -o output.png --cohere-key YOUR_KEY
    
    # Batch processing
    python main.py --batch pairs.csv --cohere-key YOUR_KEY
    
    # Directory scanning
    python main.py --directory ./images --prefix body_swap --cohere-key YOUR_KEY

Default weights [1.0, 2.5, 1.5] are optimized for body-swap analysis.
"""

import argparse
import cv2
import os
import glob
import csv
import time
from datetime import datetime

from methods.method_pixel_diff import pixel_difference_score
from methods.method_embedding  import embedding_difference_score
from methods.method_pose       import pose_difference_score
from methods.method_combine_scores    import combined_score


def process_single_pair(input_path, output_path, cohere_key, weights):
    """Process a single image pair and return results."""
    try:
        img1 = cv2.imread(input_path)
        img2 = cv2.imread(output_path)
        
        if img1 is None:
            raise ValueError(f"Could not load input image: {input_path}")
        if img2 is None:
            raise ValueError(f"Could not load output image: {output_path}")

        ps = pixel_difference_score(img1, img2)
        es = embedding_difference_score(img1, img2, cohere_key)
        os_score = pose_difference_score(img1, img2)

        w = {'pixel': weights[0], 'embedding': weights[1], 'pose': weights[2]}
        cs = combined_score(ps, es, os_score, w)

        return {
            'input_path': input_path,
            'output_path': output_path,
            'pixel_score': ps,
            'embedding_score': es,
            'pose_score': os_score,
            'combined_score': cs,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'input_path': input_path,
            'output_path': output_path,
            'pixel_score': None,
            'embedding_score': None,
            'pose_score': None,
            'combined_score': None,
            'success': False,
            'error': str(e)
        }


def find_image_pairs(directory, prefix="pair"):
    """Find all image pairs in a directory with given prefix."""
    input_pattern = os.path.join(directory, f"{prefix}_input_*.png")
    input_files = glob.glob(input_pattern)
    
    pairs = []
    for input_file in sorted(input_files):
        # Extract the number from input filename
        basename = os.path.basename(input_file)
        if basename.startswith(f"{prefix}_input_"):
            number_part = basename[len(f"{prefix}_input_"):-4]  # Remove prefix and .png
            output_file = os.path.join(directory, f"{prefix}_output_{number_part}.png")
            
            if os.path.exists(output_file):
                pairs.append((input_file, output_file))
            else:
                print(f"Warning: Missing output file for {input_file}")
    
    return pairs


def load_batch_file(batch_file):
    """Load image pairs from a batch CSV file."""
    pairs = []
    try:
        # Get the parent directory of the batch file
        batch_dir = os.path.dirname(os.path.abspath(batch_file))
        
        with open(batch_file, 'r') as f:
            reader = csv.reader(f)
            # Skip header if present
            first_row = next(reader, None)
            if first_row and (first_row[0].lower() == 'input_path' or first_row[0].lower() == 'input'):
                # Header row detected, continue with next rows
                pass
            else:
                # No header, process first row as data
                if first_row and len(first_row) >= 2:
                    input_path = os.path.join(batch_dir, first_row[0].strip())
                    output_path = os.path.join(batch_dir, first_row[1].strip())
                    if os.path.exists(input_path) and os.path.exists(output_path):
                        pairs.append((input_path, output_path))
                    else:
                        if not os.path.exists(input_path):
                            print(f"Warning: Input file not found: {input_path}")
                        if not os.path.exists(output_path):
                            print(f"Warning: Output file not found: {output_path}")
            
            # Process remaining rows
            for line_num, row in enumerate(reader, 2):
                if len(row) < 2:
                    print(f"Warning: Invalid format on line {line_num}: {row}")
                    continue
                    
                input_path = os.path.join(batch_dir, row[0].strip())
                output_path = os.path.join(batch_dir, row[1].strip())
                
                if not os.path.exists(input_path):
                    print(f"Warning: Input file not found: {input_path}")
                    continue
                if not os.path.exists(output_path):
                    print(f"Warning: Output file not found: {output_path}")
                    continue
                    
                pairs.append((input_path, output_path))
    except Exception as e:
        print(f"Error reading batch file: {e}")
        return []
    
    return pairs


def save_results_csv(results, filename):
    """Save results to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['input_path', 'output_path', 'pixel_score', 'embedding_score', 
                     'pose_score', 'combined_score', 'success', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)





def print_summary_statistics(results):
    """Print summary statistics for batch processing."""
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total pairs processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFAILED PAIRS:")
        for result in failed:
            print(f"  {result['input_path']} -> {result['output_path']}")
            print(f"    Error: {result['error']}")
    
    if successful:
        # Calculate statistics
        pixel_scores = [r['pixel_score'] for r in successful]
        embedding_scores = [r['embedding_score'] for r in successful]
        pose_scores = [r['pose_score'] for r in successful]
        combined_scores = [r['combined_score'] for r in successful]
        
        print(f"\nSTATISTICS (successful pairs):")
        print(f"  Pixel Score     - Mean: {sum(pixel_scores)/len(pixel_scores):.4f}, "
              f"Min: {min(pixel_scores):.4f}, Max: {max(pixel_scores):.4f}")
        print(f"  Embedding Score - Mean: {sum(embedding_scores)/len(embedding_scores):.4f}, "
              f"Min: {min(embedding_scores):.4f}, Max: {max(embedding_scores):.4f}")
        print(f"  Pose Score      - Mean: {sum(pose_scores)/len(pose_scores):.4f}, "
              f"Min: {min(pose_scores):.4f}, Max: {max(pose_scores):.4f}")
        print(f"  Combined Score  - Mean: {sum(combined_scores)/len(combined_scores):.4f}, "
              f"Min: {min(combined_scores):.4f}, Max: {max(combined_scores):.4f}")


def main():
    p = argparse.ArgumentParser(description='Image Similarity Analysis Tool with Batch Processing')
    
    # Single pair mode
    p.add_argument('-i', '--input', help='Path to original image')
    p.add_argument('-o', '--output', help='Path to processed image')
    
    # Batch processing mode
    p.add_argument('--batch', help='Path to batch CSV file with input_path,output_path columns')
    p.add_argument('--directory', help='Directory containing image pairs')
    p.add_argument('--prefix', default='pair', help='Filename prefix for directory mode (default: pair)')
    
    # Common options
    p.add_argument('-w', '--weights', nargs=3, type=float,
                   default=[1.0, 1.0, 1.0],
                   help='Weights: pixel, embedding, pose (default: 1.0 1.0 1.0)')
    p.add_argument('--cohere-key', required=True, help='Cohere API key')
    
    # Output options
    p.add_argument('--output-csv', help='Save results to specified CSV file')
    p.add_argument('--verbose', '-v', action='store_true', help='Verbose output for each pair')
    
    args = p.parse_args()

    # Validate arguments
    mode_count = sum([
        bool(args.input and args.output),
        bool(args.batch),
        bool(args.directory)
    ])
    
    if mode_count != 1:
        print("Error: Specify exactly one mode:")
        print("  Single pair: --input and --output")
        print("  Batch CSV:   --batch")
        print("  Directory:   --directory")
        p.print_help()
        return

    # Determine processing mode and get image pairs
    pairs = []
    
    if args.input and args.output:
        # Single pair mode
        pairs = [(args.input, args.output)]
        print(f"Processing single pair: {args.input} -> {args.output}")
        
    elif args.batch:
        # Batch CSV mode
        pairs = load_batch_file(args.batch)
        print(f"Loaded {len(pairs)} pairs from batch CSV: {args.batch}")
        
    elif args.directory:
        # Directory mode
        pairs = find_image_pairs(args.directory, args.prefix)
        print(f"Found {len(pairs)} pairs in directory: {args.directory}")

    if not pairs:
        print("No image pairs to process!")
        return

    # Process all pairs
    results = []
    weights = args.weights
    
    print(f"\nUsing weights: Pixel={weights[0]}, Embedding={weights[1]}, Pose={weights[2]}")
    print(f"Processing {len(pairs)} pairs...\n")
    
    for i, (input_path, output_path) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] Processing: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
        
        result = process_single_pair(input_path, output_path, args.cohere_key, weights)
        results.append(result)
        
        if result['success']:
            if args.verbose or len(pairs) == 1:
                print(f"  Pixel Score:     {result['pixel_score']:.4f}")
                print(f"  Embedding Score: {result['embedding_score']:.4f}")
                print(f"  Pose Score:      {result['pose_score']:.4f}")
                print(f"  Combined Score:  {result['combined_score']:.4f}")
            else:
                print(f"  ✓ Combined Score: {result['combined_score']:.4f}")
        else:
            print(f"  ✗ Error: {result['error']}")
        
        # Add delay between Cohere API calls to avoid rate limiting (except for the last pair)
        if len(pairs) > 1 and i < len(pairs) and result['success']:
            print(f"  Waiting 1 seconds to avoid rate limits...")
            time.sleep(1)
            
        print()

    # Save results to CSV file
    if args.output_csv:
        save_results_csv(results, args.output_csv)
        print(f"Results saved to: {args.output_csv}")
    elif len(pairs) > 1:
        # Auto-save for batch processing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_csv = f"similarity_results_{timestamp}.csv"
        
        save_results_csv(results, default_csv)
        print(f"Results saved to: {default_csv}")

    # Print summary for batch processing
    if len(pairs) > 1:
        print_summary_statistics(results)


if __name__ == '__main__':
    main()