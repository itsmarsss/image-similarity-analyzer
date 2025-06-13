"""
main.py

Comprehensive image similarity analysis tool with flexible method selection and batch processing.
Analyzes similarity between original and processed images using multiple computer vision
and AI techniques: pixel differences, semantic embeddings, and pose detection.

Features:
- Single pair analysis with detailed scoring
- Batch processing from CSV files with auto-save
- Directory scanning for image pairs with custom prefixes
- Selective method execution (disable pixel/embedding/pose analysis)
- Flexible weighted scoring system with individual or combined weights
- Rate-limited API calls with configurable delays
- Multiple output modes (verbose, normal, quiet)
- CSV export with method tracking and comprehensive statistics

Analysis Methods:
- Pixel Difference: Direct pixel-level comparison using computer vision
- Semantic Embedding: AI-powered semantic similarity via Cohere API
- Pose Detection: Human pose comparison using MediaPipe

Usage Examples:
    # Single pair analysis
    python main.py -i input.png -o output.png --cohere-key YOUR_KEY
    
    # Batch processing with all methods
    python main.py --batch pairs.csv --cohere-key YOUR_KEY
    
    # Directory scanning with custom prefix
    python main.py --directory ./images --prefix body_swap --cohere-key YOUR_KEY
    
    # Disable expensive embedding analysis
    python main.py --batch pairs.csv --disable-embedding
    
    # Custom weights (individual)
    python main.py --batch pairs.csv --pixel-weight 2.0 --embedding-weight 1.5 --pose-weight 0.5 --cohere-key YOUR_KEY
    
    # Custom weights (combined)
    python main.py --batch pairs.csv -w 2.0 1.5 0.5 --cohere-key YOUR_KEY
    
    # Quiet mode with custom rate limiting
    python main.py --batch pairs.csv --quiet --rate-limit-delay 2.0 --cohere-key YOUR_KEY
    
    # Only pixel and pose analysis (no API required)
    python main.py --batch pairs.csv --disable-embedding --verbose

Method Control:
- Use --disable-pixel, --disable-embedding, or --disable-pose to skip specific analyses
- At least one method must remain enabled
- Cohere API key only required when embedding analysis is enabled
- Disabled methods automatically get zero weight in final scoring

Output Options:
- --verbose: Detailed scores for each method and pair
- --quiet: Minimal output (errors and final results only)
- --output-csv: Save to specific CSV file
- --no-auto-save: Disable automatic timestamped CSV generation

Default weights [1.0, 1.0, 1.0] provide equal weighting. Adjust based on your use case.
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


def process_single_pair(input_path, output_path, cohere_key, weights, disabled_methods=None):
    """Process a single image pair and return results."""
    if disabled_methods is None:
        disabled_methods = {'pixel': False, 'embedding': False, 'pose': False}
    
    try:
        img1 = cv2.imread(input_path)
        img2 = cv2.imread(output_path)
        
        if img1 is None:
            raise ValueError(f"Could not load input image: {input_path}")
        if img2 is None:
            raise ValueError(f"Could not load output image: {output_path}")

        # Initialize scores
        ps = None
        es = None
        os_score = None

        # Run enabled methods only
        if not disabled_methods['pixel']:
            ps = pixel_difference_score(img1, img2)
        
        if not disabled_methods['embedding']:
            es = embedding_difference_score(img1, img2, cohere_key)
        
        if not disabled_methods['pose']:
            os_score = pose_difference_score(img1, img2)

        # Calculate combined score (handles None values)
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
            'error': None,
            'methods_used': {
                'pixel': not disabled_methods['pixel'],
                'embedding': not disabled_methods['embedding'],
                'pose': not disabled_methods['pose']
            }
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
            'error': str(e),
            'methods_used': {
                'pixel': not disabled_methods['pixel'],
                'embedding': not disabled_methods['embedding'],
                'pose': not disabled_methods['pose']
            }
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
                     'pose_score', 'combined_score', 'success', 'error',
                     'methods_pixel', 'methods_embedding', 'methods_pose']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            # Flatten methods_used into separate columns
            row = result.copy()
            if 'methods_used' in row:
                row['methods_pixel'] = row['methods_used']['pixel']
                row['methods_embedding'] = row['methods_used']['embedding']
                row['methods_pose'] = row['methods_used']['pose']
                del row['methods_used']
            writer.writerow(row)





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
    
    # Input/Output arguments (consistent with helper files)
    p.add_argument('--input-video', help='Path to input video file')
    p.add_argument('--output-video', help='Path to output video file')
    p.add_argument('-i', '--input', help='Path to original image (single pair mode)')
    p.add_argument('-o', '--output', help='Path to processed image (single pair mode)')
    
    # Batch processing mode
    p.add_argument('--batch', help='Path to batch CSV file with input_path,output_path columns')
    p.add_argument('--directory', help='Directory containing image pairs')
    p.add_argument('--prefix', default='pair', help='Filename prefix for directory mode (default: pair)')
    
    # Method control options
    p.add_argument('--disable-pixel', action='store_true', 
                   help='Disable pixel difference analysis')
    p.add_argument('--disable-embedding', action='store_true', 
                   help='Disable semantic embedding analysis')
    p.add_argument('--disable-pose', action='store_true', 
                   help='Disable pose detection analysis')
    
    # Scoring options
    p.add_argument('-w', '--weights', nargs=3, type=float,
                   default=[1.0, 1.0, 1.0],
                   help='Weights: pixel, embedding, pose (default: 1.0 1.0 1.0)')
    p.add_argument('--pixel-weight', type=float, default=1.0,
                   help='Weight for pixel difference score (default: 1.0)')
    p.add_argument('--embedding-weight', type=float, default=1.0,
                   help='Weight for embedding similarity score (default: 1.0)')
    p.add_argument('--pose-weight', type=float, default=1.0,
                   help='Weight for pose similarity score (default: 1.0)')
    
    # API configuration
    p.add_argument('--cohere-key', help='Cohere API key for embedding analysis (required unless --disable-embedding is used)')
    p.add_argument('--rate-limit-delay', type=float, default=1.0,
                   help='Delay between API calls in seconds (default: 1.0)')
    
    # Output options
    p.add_argument('--output-csv', help='Save results to specified CSV file')
    p.add_argument('--auto-save', action='store_true', default=True,
                   help='Automatically save batch results to timestamped CSV (default: True)')
    p.add_argument('--no-auto-save', action='store_true',
                   help='Disable automatic saving of batch results')
    p.add_argument('--verbose', '-v', action='store_true', help='Verbose output for each pair')
    p.add_argument('--quiet', '-q', action='store_true', help='Minimal output (only errors and final results)')
    
    args = p.parse_args()

    # Handle weight arguments (prefer individual weights over combined weights)
    if any([args.pixel_weight != 1.0, args.embedding_weight != 1.0, args.pose_weight != 1.0]):
        weights = [args.pixel_weight, args.embedding_weight, args.pose_weight]
    else:
        weights = args.weights
    
    # Handle auto-save logic
    auto_save = args.auto_save and not args.no_auto_save
    
    # Validate method selection
    methods_disabled = [args.disable_pixel, args.disable_embedding, args.disable_pose]
    if all(methods_disabled):
        print("Error: Cannot disable all analysis methods. At least one method must be enabled.")
        return
    
    # Adjust weights for disabled methods
    if args.disable_pixel:
        weights[0] = 0.0
    if args.disable_embedding:
        weights[1] = 0.0
    if args.disable_pose:
        weights[2] = 0.0
    
    # Validate Cohere key requirement
    if not args.disable_embedding and not args.cohere_key:
        print("Error: --cohere-key is required when embedding analysis is enabled")
        print("Use --disable-embedding to skip embedding analysis, or provide --cohere-key")
        return

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
    
    # Create disabled methods dictionary
    disabled_methods = {
        'pixel': args.disable_pixel,
        'embedding': args.disable_embedding,
        'pose': args.disable_pose
    }
    
    # Display configuration
    if not args.quiet:
        print(f"\nConfiguration:")
        print(f"  Weights: Pixel={weights[0]}, Embedding={weights[1]}, Pose={weights[2]}")
        print(f"  Methods: Pixel={'✗' if args.disable_pixel else '✓'}, "
              f"Embedding={'✗' if args.disable_embedding else '✓'}, "
              f"Pose={'✗' if args.disable_pose else '✓'}")
        print(f"  Rate limit delay: {args.rate_limit_delay}s")
        print(f"Processing {len(pairs)} pairs...\n")
    
    for i, (input_path, output_path) in enumerate(pairs, 1):
        if not args.quiet:
            print(f"[{i}/{len(pairs)}] Processing: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
        
        result = process_single_pair(input_path, output_path, args.cohere_key, weights, disabled_methods)
        results.append(result)
        
        if result['success']:
            if args.verbose or len(pairs) == 1:
                if result['pixel_score'] is not None:
                    print(f"  Pixel Score:     {result['pixel_score']:.4f}")
                if result['embedding_score'] is not None:
                    print(f"  Embedding Score: {result['embedding_score']:.4f}")
                if result['pose_score'] is not None:
                    print(f"  Pose Score:      {result['pose_score']:.4f}")
                print(f"  Combined Score:  {result['combined_score']:.4f}")
            elif not args.quiet:
                print(f"  ✓ Combined Score: {result['combined_score']:.4f}")
        else:
            if not args.quiet:
                print(f"  ✗ Error: {result['error']}")
        
        # Add delay between API calls to avoid rate limiting (except for the last pair)
        if len(pairs) > 1 and i < len(pairs) and result['success'] and not args.disable_embedding:
            if not args.quiet:
                print(f"  Waiting {args.rate_limit_delay} seconds to avoid rate limits...")
            time.sleep(args.rate_limit_delay)
            
        if not args.quiet:
            print()

    # Save results to CSV file
    if args.output_csv:
        save_results_csv(results, args.output_csv)
        if not args.quiet:
            print(f"Results saved to: {args.output_csv}")
    elif len(pairs) > 1 and auto_save:
        # Auto-save for batch processing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_csv = f"similarity_results_{timestamp}.csv"
        
        save_results_csv(results, default_csv)
        if not args.quiet:
            print(f"Results saved to: {default_csv}")

    # Print summary for batch processing
    if len(pairs) > 1 and not args.quiet:
        print_summary_statistics(results)


if __name__ == '__main__':
    main()