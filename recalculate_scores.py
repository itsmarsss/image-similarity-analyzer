#!/usr/bin/env python3
"""
recalculate_scores.py

Recalculates combined scores from similarity analysis results with configurable 
scale factors and weights. Scale factors are applied first to individual scores,
then weighted combination is performed.

Usage:
    python recalculate_scores.py input.csv [options]
    
    --pixel-scale FLOAT     Scale factor for pixel scores (default: 1.0)
    --embedding-scale FLOAT Scale factor for embedding scores (default: 1.0)  
    --pose-scale FLOAT      Scale factor for pose scores (default: 1.0)
    --pixel-weight FLOAT    Weight for pixel scores (default: 1.0)
    --embedding-weight FLOAT Weight for embedding scores (default: 1.0)
    --pose-weight FLOAT     Weight for pose scores (default: 1.0)
    --output OUTPUT_FILE    Output CSV file (default: recalculated_results.csv)
    --verbose              Show detailed statistics
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import os


def recalculate_combined_score(pixel_score, embedding_score, pose_score, 
                              pixel_scale, embedding_scale, pose_scale,
                              pixel_weight, embedding_weight, pose_weight):
    """
    Recalculate combined score with scale factors and weights.
    
    Process:
    1. Apply scale factors: scaled_score = original_score * scale_factor
    2. Apply weights: combined = (scaled_pixel * pixel_weight + 
                                 scaled_embedding * embedding_weight + 
                                 scaled_pose * pose_weight) / total_weight
    """
    # Apply scale factors first
    scaled_pixel = pixel_score * pixel_scale
    scaled_embedding = embedding_score * embedding_scale
    scaled_pose = pose_score * pose_scale
    
    # Calculate total weight for normalization
    total_weight = pixel_weight + embedding_weight + pose_weight
    
    # Apply weighted combination
    combined = (scaled_pixel * pixel_weight + 
                scaled_embedding * embedding_weight + 
                scaled_pose * pose_weight) / total_weight
    
    return combined, scaled_pixel, scaled_embedding, scaled_pose


def print_statistics(df, original_col, new_col, verbose=False):
    """Print comparison statistics between original and new scores."""
    print(f"\n{'='*60}")
    print(f"SCORE RECALCULATION STATISTICS")
    print(f"{'='*60}")
    
    # Filter out failed rows
    successful_df = df[df['success'] == True].copy()
    
    if len(successful_df) == 0:
        print("No successful results to analyze!")
        return
    
    original_scores = successful_df[original_col]
    new_scores = successful_df[new_col]
    
    print(f"Successfully processed pairs: {len(successful_df)}")
    print(f"Failed pairs: {len(df) - len(successful_df)}")
    
    print(f"\nORIGINAL {original_col.upper()}:")
    print(f"  Mean: {original_scores.mean():.6f}")
    print(f"  Min:  {original_scores.min():.6f}")
    print(f"  Max:  {original_scores.max():.6f}")
    print(f"  Std:  {original_scores.std():.6f}")
    
    print(f"\nNEW {new_col.upper()}:")
    print(f"  Mean: {new_scores.mean():.6f}")
    print(f"  Min:  {new_scores.min():.6f}")
    print(f"  Max:  {new_scores.max():.6f}")
    print(f"  Std:  {new_scores.std():.6f}")
    
    # Calculate change statistics
    absolute_change = new_scores - original_scores
    relative_change = ((new_scores - original_scores) / original_scores) * 100
    
    print(f"\nCHANGE STATISTICS:")
    print(f"  Mean absolute change: {absolute_change.mean():.6f}")
    print(f"  Mean relative change: {relative_change.mean():.2f}%")
    print(f"  Max increase: {absolute_change.max():.6f}")
    print(f"  Max decrease: {absolute_change.min():.6f}")
    
    if verbose:
        # Calculate change for sorting
        successful_df['change'] = successful_df[new_col] - successful_df[original_col]
        
        print(f"\nTOP 10 LARGEST INCREASES:")
        top_increases = successful_df.nlargest(10, 'change')
        for idx, row in top_increases.iterrows():
            change = row['change']
            print(f"  {os.path.basename(row['input_path'])}: {row[original_col]:.6f} → {row[new_col]:.6f} (+{change:.6f})")
        
        print(f"\nTOP 10 LARGEST DECREASES:")
        top_decreases = successful_df.nsmallest(10, 'change')
        for idx, row in top_decreases.iterrows():
            change = row['change']
            print(f"  {os.path.basename(row['input_path'])}: {row[original_col]:.6f} → {row[new_col]:.6f} ({change:.6f})")


def main():
    parser = argparse.ArgumentParser(description='Recalculate combined scores with scale factors and weights')
    
    # Input/Output
    parser.add_argument('input_csv', help='Input CSV file with similarity results')
    parser.add_argument('--output', '-o', default='recalculated_results.csv', 
                       help='Output CSV file (default: recalculated_results.csv)')
    
    # Scale factors (applied first)
    parser.add_argument('--pixel-scale', type=float, default=1.0,
                       help='Scale factor for pixel scores (default: 1.0)')
    parser.add_argument('--embedding-scale', type=float, default=1.0,
                       help='Scale factor for embedding scores (default: 1.0)')
    parser.add_argument('--pose-scale', type=float, default=1.0,
                       help='Scale factor for pose scores (default: 1.0)')
    
    # Weights (applied after scaling)
    parser.add_argument('--pixel-weight', type=float, default=1.0,
                       help='Weight for pixel scores (default: 1.0)')
    parser.add_argument('--embedding-weight', type=float, default=1.0,
                       help='Weight for embedding scores (default: 1.0)')
    parser.add_argument('--pose-weight', type=float, default=1.0,
                       help='Weight for pose scores (default: 1.0)')
    
    # Options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed statistics and top changes')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file '{args.input_csv}' not found!")
        return
    
    print(f"Loading similarity results from: {args.input_csv}")
    
    try:
        # Read the CSV
        df = pd.read_csv(args.input_csv)
        
        # Validate required columns
        required_cols = ['pixel_score', 'embedding_score', 'pose_score', 'combined_score', 'success']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return
        
        print(f"Loaded {len(df)} rows from CSV")
        
        # Display configuration
        print(f"\nCONFIGURATION:")
        print(f"  Scale Factors:")
        print(f"    Pixel:     {args.pixel_scale}")
        print(f"    Embedding: {args.embedding_scale}")
        print(f"    Pose:      {args.pose_scale}")
        print(f"  Weights:")
        print(f"    Pixel:     {args.pixel_weight}")
        print(f"    Embedding: {args.embedding_weight}")
        print(f"    Pose:      {args.pose_weight}")
        
        # Recalculate scores for successful rows
        new_combined_scores = []
        scaled_pixel_scores = []
        scaled_embedding_scores = []
        scaled_pose_scores = []
        
        for idx, row in df.iterrows():
            if row['success'] and pd.notna(row['pixel_score']) and pd.notna(row['embedding_score']) and pd.notna(row['pose_score']):
                new_combined, scaled_pixel, scaled_embedding, scaled_pose = recalculate_combined_score(
                    row['pixel_score'], row['embedding_score'], row['pose_score'],
                    args.pixel_scale, args.embedding_scale, args.pose_scale,
                    args.pixel_weight, args.embedding_weight, args.pose_weight
                )
                new_combined_scores.append(new_combined)
                scaled_pixel_scores.append(scaled_pixel)
                scaled_embedding_scores.append(scaled_embedding)
                scaled_pose_scores.append(scaled_pose)
            else:
                # Keep NaN for failed rows
                new_combined_scores.append(np.nan)
                scaled_pixel_scores.append(np.nan)
                scaled_embedding_scores.append(np.nan)
                scaled_pose_scores.append(np.nan)
        
        # Add new columns to dataframe
        df['scaled_pixel_score'] = scaled_pixel_scores
        df['scaled_embedding_score'] = scaled_embedding_scores
        df['scaled_pose_score'] = scaled_pose_scores
        df['new_combined_score'] = new_combined_scores
        df['original_combined_score'] = df['combined_score']  # Keep original for comparison
        
        # Replace the combined_score column with new values
        df['combined_score'] = df['new_combined_score']
        
        # Add metadata columns
        df['recalculation_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df['pixel_scale_factor'] = args.pixel_scale
        df['embedding_scale_factor'] = args.embedding_scale
        df['pose_scale_factor'] = args.pose_scale
        df['pixel_weight'] = args.pixel_weight
        df['embedding_weight'] = args.embedding_weight
        df['pose_weight'] = args.pose_weight
        
        # Save results
        df.to_csv(args.output, index=False)
        print(f"\nRecalculated results saved to: {args.output}")
        
        # Print statistics
        print_statistics(df, 'original_combined_score', 'combined_score', args.verbose)
        
        # Print summary of changes
        successful_df = df[df['success'] == True]
        if len(successful_df) > 0:
            increases = (successful_df['combined_score'] > successful_df['original_combined_score']).sum()
            decreases = (successful_df['combined_score'] < successful_df['original_combined_score']).sum()
            unchanged = (successful_df['combined_score'] == successful_df['original_combined_score']).sum()
            
            print(f"\nSUMMARY OF CHANGES:")
            print(f"  Scores increased: {increases}")
            print(f"  Scores decreased: {decreases}")
            print(f"  Scores unchanged: {unchanged}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return


if __name__ == '__main__':
    main() 