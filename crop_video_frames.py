#!/usr/bin/env python3
"""
crop_video_frames.py

Extract and crop video frames based on bounding box data from bounding_data.json.
Each frame is cropped according to the bounding box coordinates and saved 
as individual image files.

Usage:
    python crop_video_frames.py video_file.mp4 [options]
    
    --json-file JSON_FILE        Path to bounding_data.json file (default: bounding_data.json)
    --output-dir OUTPUT_DIR      Output directory for cropped frames (default: cropped_frames)
    --bbox-index INDEX           Which bounding box to use if multiple (default: 0)
    --frame-prefix PREFIX        Prefix for output frame files (default: frame)
    --format FORMAT              Output image format: png, jpg (default: png)
    --skip-missing               Skip frames without bounding box data
    --verbose                    Show detailed progress information
"""

import argparse
import cv2
import json
import os
import sys
from pathlib import Path


def load_bounding_data(json_file):
    """Load bounding box data from JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data.get('boundingData', [])
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found!")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file - {e}")
        return None


def create_frame_lookup(bounding_data):
    """Create a lookup dictionary for frame index to bounding box data."""
    frame_lookup = {}
    for frame_data in bounding_data:
        frame_idx = frame_data.get('frameIndex')
        bounding_list = frame_data.get('boundingList', [])
        if frame_idx is not None and bounding_list:
            frame_lookup[frame_idx] = bounding_list
    return frame_lookup


def normalize_to_pixel_coords(bbox, frame_width, frame_height):
    """Convert normalized coordinates (0-1) to pixel coordinates."""
    start_x = int(bbox['startX'] * frame_width)
    start_y = int(bbox['startY'] * frame_height)
    end_x = int(bbox['endX'] * frame_width)
    end_y = int(bbox['endY'] * frame_height)
    
    # Ensure coordinates are within frame bounds
    start_x = max(0, min(start_x, frame_width - 1))
    start_y = max(0, min(start_y, frame_height - 1))
    end_x = max(start_x + 1, min(end_x, frame_width))
    end_y = max(start_y + 1, min(end_y, frame_height))
    
    return start_x, start_y, end_x, end_y


def crop_frame(frame, bbox, frame_width, frame_height):
    """Crop frame using bounding box coordinates."""
    start_x, start_y, end_x, end_y = normalize_to_pixel_coords(bbox, frame_width, frame_height)
    
    # Crop the frame
    cropped = frame[start_y:end_y, start_x:end_x]
    
    return cropped, (start_x, start_y, end_x, end_y)


def process_video(video_path, json_file, output_dir, bbox_index=0, frame_prefix="frame", 
                 output_format="png", skip_missing=False, verbose=False):
    """Process video and save cropped frames."""
    
    # Load bounding box data
    if verbose:
        print(f"Loading bounding box data from: {json_file}")
    
    bounding_data = load_bounding_data(json_file)
    if bounding_data is None:
        return False
    
    frame_lookup = create_frame_lookup(bounding_data)
    if verbose:
        print(f"Loaded bounding data for {len(frame_lookup)} frames")
    
    # Open video
    if verbose:
        print(f"Opening video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if verbose:
        print(f"Video properties:")
        print(f"  Total frames: {total_frames}")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {fps:.2f}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process frames
    frame_count = 0
    saved_count = 0
    skipped_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if verbose and frame_count % 50 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")
        
        # Check if we have bounding box data for this frame
        if frame_count in frame_lookup:
            bounding_list = frame_lookup[frame_count]
            
            # Handle multiple bounding boxes
            if bbox_index >= len(bounding_list):
                if not skip_missing:
                    print(f"Warning: Frame {frame_count} has {len(bounding_list)} bounding boxes, "
                          f"but index {bbox_index} requested. Using index 0.")
                    bbox = bounding_list[0]
                else:
                    skipped_count += 1
                    frame_count += 1
                    continue
            else:
                bbox = bounding_list[bbox_index]
            
            # Crop frame
            try:
                cropped_frame, coords = crop_frame(frame, bbox, frame_width, frame_height)
                
                # Save cropped frame
                output_filename = f"{frame_prefix}_{frame_count:06d}.{output_format}"
                output_file_path = output_path / output_filename
                
                success = cv2.imwrite(str(output_file_path), cropped_frame)
                if success:
                    saved_count += 1
                    if verbose:
                        print(f"Saved {output_filename} - Crop: ({coords[0]}, {coords[1]}) to ({coords[2]}, {coords[3]}) - Size: {cropped_frame.shape[1]}x{cropped_frame.shape[0]}")
                else:
                    print(f"Error: Failed to save {output_filename}")
                    
            except Exception as e:
                print(f"Error cropping frame {frame_count}: {e}")
                skipped_count += 1
        else:
            # No bounding box data for this frame
            if not skip_missing:
                print(f"Warning: No bounding box data for frame {frame_count}")
            skipped_count += 1
        
        frame_count += 1
    
    cap.release()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames saved: {saved_count}")
    print(f"Frames skipped: {skipped_count}")
    print(f"Output directory: {output_dir}")
    print(f"Output format: {output_format}")
    
    if saved_count > 0:
        print(f"\nSample output files:")
        for i, file in enumerate(sorted(output_path.glob(f"{frame_prefix}_*.{output_format}"))[:5]):
            print(f"  {file.name}")
        if saved_count > 5:
            print(f"  ... and {saved_count - 5} more files")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Extract and crop video frames based on bounding box data')
    
    # Required argument
    parser.add_argument('video_file', help='Input video file path')
    
    # Optional arguments
    parser.add_argument('--json-file', default='rleBound.json',
                       help='Path to rleBound.json file (default: rleBound.json)')
    parser.add_argument('--output-dir', default='cropped_frames',
                       help='Output directory for cropped frames (default: cropped_frames)')
    parser.add_argument('--bbox-index', type=int, default=0,
                       help='Which bounding box to use if multiple per frame (default: 0)')
    parser.add_argument('--frame-prefix', default='frame',
                       help='Prefix for output frame files (default: frame)')
    parser.add_argument('--format', choices=['png', 'jpg'], default='png',
                       help='Output image format (default: png)')
    parser.add_argument('--skip-missing', action='store_true',
                       help='Skip frames without bounding box data (default: warn but continue)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed progress information')
    
    args = parser.parse_args()
    
    # Validate input video file
    if not os.path.exists(args.video_file):
        print(f"Error: Video file '{args.video_file}' not found!")
        return 1
    
    # Validate JSON file
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file '{args.json_file}' not found!")
        return 1
    
    print(f"Starting video frame extraction and cropping...")
    print(f"Video file: {args.video_file}")
    print(f"JSON file: {args.json_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Bounding box index: {args.bbox_index}")
    print(f"Frame prefix: {args.frame_prefix}")
    print(f"Output format: {args.format}")
    print(f"Skip missing: {args.skip_missing}")
    
    success = process_video(
        video_path=args.video_file,
        json_file=args.json_file,
        output_dir=args.output_dir,
        bbox_index=args.bbox_index,
        frame_prefix=args.frame_prefix,
        output_format=args.format,
        skip_missing=args.skip_missing,
        verbose=args.verbose
    )
    
    if success:
        print("\n✓ Video processing completed successfully!")
        return 0
    else:
        print("\n✗ Video processing failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 