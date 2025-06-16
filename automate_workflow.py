#!/usr/bin/env python3
"""
automate_workflow.py

Short automation script for complete video analysis workflow:
1. Create full and cropped directories
2. Extract full frames from input/output videos
3. Crop frames using bounding_data.json
4. Run similarity analysis on both datasets
5. Launch result viewers on different ports

Usage:
    python automate_workflow.py --input input.mp4 --output output.mp4 --bounds bounding_data.json --cohere-key YOUR_KEY
"""

import argparse
import os
import subprocess
import sys
import time
import cv2
from pathlib import Path


def run_command(cmd, description, background=False):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        if background:
            # Run in background and return process
            return subprocess.Popen(cmd)
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ {description} completed successfully")
            return result
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return None
    except FileNotFoundError as e:
        print(f"✗ {description} failed:")
        print(f"Error: Command not found - {e}")
        return None


def extract_frames_opencv(video_path, output_dir, prefix, frame_interval=10):
    """Extract frames using OpenCV as fallback when ffmpeg is not available"""
    print(f"Using OpenCV to extract frames from {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every nth frame
        if frame_count % frame_interval == 0:
            output_filename = f"{prefix}_{saved_count:03d}.png"
            output_path = output_dir / output_filename
            
            if cv2.imwrite(str(output_path), frame):
                saved_count += 1
            else:
                print(f"Error: Failed to save frame {output_filename}")
        
        frame_count += 1
    
    cap.release()
    print(f"✓ Extracted {saved_count} frames from {frame_count} total frames")
    return True



def main():
    parser = argparse.ArgumentParser(description="Automated video analysis workflow")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output video file")
    parser.add_argument("--bounds", required=True, help="Bounding data JSON file")
    parser.add_argument("--cohere-key", help="Cohere API key for embedding analysis")
    parser.add_argument("--frame-interval", type=int, default=10, help="Frame extraction interval")
    
    # Method control options
    parser.add_argument("--disable-pixel", action="store_true", help="Disable pixel analysis")
    parser.add_argument("--disable-embedding", action="store_true", help="Disable embedding analysis")
    parser.add_argument("--disable-pose", action="store_true", help="Disable pose analysis")
    
    args = parser.parse_args()
    
    # Validate method selection
    if args.disable_pixel and args.disable_embedding and args.disable_pose:
        print("Error: Cannot disable all analysis methods. At least one method must be enabled.")
        return 1
    
    # Validate API key requirement
    if not args.disable_embedding and not args.cohere_key:
        print("Error: --cohere-key required when embedding analysis is enabled")
        print("Use --disable-embedding to skip embedding analysis, or provide your Cohere API key")
        return 1
    
    # Validate input files
    for file_path, name in [(args.input, "Input video"), (args.output, "Output video"), (args.bounds, "Bounding data")]:
        if not os.path.exists(file_path):
            print(f"Error: {name} not found: {file_path}")
            return 1
    
    print("🚀 Starting automated video analysis workflow")
    print("=" * 50)
    
    # Step 1: Create directories
    print("\n📁 Step 1: Creating directories")
    full_dir = Path("full")
    cropped_dir = Path("cropped")
    
    full_dir.mkdir(exist_ok=True)
    cropped_dir.mkdir(exist_ok=True)
    print(f"✓ Created directories: {full_dir}, {cropped_dir}")
    
    # Step 2: Extract full frames
    print("\n🎬 Step 2: Extracting full frames using OpenCV")
    
    # Extract input frames using OpenCV
    if not extract_frames_opencv(args.input, full_dir, "pair_input", args.frame_interval):
        return 1
    
    # Extract output frames using OpenCV
    if not extract_frames_opencv(args.output, full_dir, "pair_output", args.frame_interval):
        return 1
    
    # Step 3: Crop frames using bounding data
    print("\n✂️  Step 3: Cropping frames with bounding data")
    
    # Crop input frames
    cmd = ["python", "crop_video_frames.py", args.input, "--json-file", args.bounds,
           "--output-dir", str(cropped_dir), "--frame-prefix", "pair_input", "--format", "png"]
    
    if not run_command(cmd, "Cropping input video frames"):
        return 1
    
    # Crop output frames
    cmd = ["python", "crop_video_frames.py", args.output, "--json-file", args.bounds,
           "--output-dir", str(cropped_dir), "--frame-prefix", "pair_output", "--format", "png"]
    
    if not run_command(cmd, "Cropping output video frames"):
        return 1
    
    # Step 4: Run similarity analysis on both datasets
    print("\n🔍 Step 4: Running similarity analysis")
    
    # Analyze full frames
    cmd = ["python", "main.py", "--directory", str(full_dir), "--prefix", "pair"]
    
    # Add method controls
    if args.disable_pixel:
        cmd.append("--disable-pixel")
    if args.disable_embedding:
        cmd.append("--disable-embedding")
    if args.disable_pose:
        cmd.append("--disable-pose")
    
    # Add API key if embedding is enabled
    if not args.disable_embedding and args.cohere_key:
        cmd.extend(["--cohere-key", args.cohere_key])
    
    cmd.extend(["--output-csv", "full_results.csv"])
    
    full_analysis_success = run_command(cmd, "Analyzing full frames")
    if not full_analysis_success:
        print("⚠️  Full frames analysis failed, but continuing...")
    
    # Check what files actually exist in cropped directory
    cropped_files = list(cropped_dir.glob("*.png"))
    if not cropped_files:
        print("❌ No cropped frames found, skipping cropped analysis")
        cropped_analysis_success = False
    else:
        print(f"Found {len(cropped_files)} cropped frames")
        # Check the actual naming pattern
        sample_file = cropped_files[0].name
        print(f"Sample cropped file: {sample_file}")
        
        # Try to determine the correct prefix from existing files
        if sample_file.startswith("pair_input_"):
            cropped_prefix = "pair"
        elif sample_file.startswith("all_input_"):
            cropped_prefix = "all"
        else:
            # Create a batch CSV file instead
            print("Creating batch CSV for cropped frames analysis...")
            batch_csv = "cropped_batch.csv"
            
            # Find matching input/output pairs
            input_files = sorted([f for f in cropped_files if "input" in f.name])
            output_files = sorted([f for f in cropped_files if "output" in f.name])
            
            if len(input_files) == len(output_files):
                with open(batch_csv, 'w') as f:
                    f.write("input_path,output_path\n")
                    for inp, out in zip(input_files, output_files):
                        f.write(f"{inp.name},{out.name}\n")
                
                cmd = ["python", "main.py", "--batch", batch_csv]
                
                # Add method controls
                if args.disable_pixel:
                    cmd.append("--disable-pixel")
                if args.disable_embedding:
                    cmd.append("--disable-embedding")
                if args.disable_pose:
                    cmd.append("--disable-pose")
                
                # Add API key if embedding is enabled
                if not args.disable_embedding and args.cohere_key:
                    cmd.extend(["--cohere-key", args.cohere_key])
                
                cmd.extend(["--output-csv", "cropped_results.csv"])
            else:
                print(f"❌ Mismatch: {len(input_files)} input files vs {len(output_files)} output files")
                cropped_analysis_success = False
                cmd = None
        
        if 'cropped_prefix' in locals():
            # Use directory mode with detected prefix
            cmd = ["python", "main.py", "--directory", str(cropped_dir), "--prefix", cropped_prefix]
            
            # Add method controls
            if args.disable_pixel:
                cmd.append("--disable-pixel")
            if args.disable_embedding:
                cmd.append("--disable-embedding")
            if args.disable_pose:
                cmd.append("--disable-pose")
            
            # Add API key if embedding is enabled
            if not args.disable_embedding and args.cohere_key:
                cmd.extend(["--cohere-key", args.cohere_key])
            
            cmd.extend(["--output-csv", "cropped_results.csv"])
        
        if cmd:
            cropped_analysis_success = run_command(cmd, "Analyzing cropped frames")
            if not cropped_analysis_success:
                print("⚠️  Cropped frames analysis failed")
        else:
            cropped_analysis_success = False
    
    # Step 5: Launch result viewers on different ports
    print("\n📊 Step 5: Launching result viewers")
    
    viewers = []
    
    # Launch viewer for full results on port 7860 if available
    if os.path.exists("full_results.csv"):
        cmd = ["python", "results_viewer.py", "-f", "full_results.csv", "-p", "7860"]
        full_viewer = run_command(cmd, "Launching full results viewer (port 7860)", background=True)
        if full_viewer:
            viewers.append(("Full frames", "http://localhost:7860", full_viewer))
    else:
        print("⚠️  full_results.csv not found, skipping full results viewer")
    
    # Launch viewer for cropped results on port 7861 if available
    if os.path.exists("cropped_results.csv"):
        cmd = ["python", "results_viewer.py", "-f", "cropped_results.csv", "-p", "7861"]
        cropped_viewer = run_command(cmd, "Launching cropped results viewer (port 7861)", background=True)
        if cropped_viewer:
            viewers.append(("Cropped frames", "http://localhost:7861", cropped_viewer))
    else:
        print("⚠️  cropped_results.csv not found, skipping cropped results viewer")
    
    # Give viewers time to start
    if viewers:
        time.sleep(3)
    
    print("\n🎉 Workflow completed!")
    print("=" * 50)
    
    if viewers:
        print("📊 Active Result Viewers:")
        for name, url, _ in viewers:
            print(f"  {name}: {url}")
    else:
        print("❌ No result viewers launched (no CSV files found)")
    
    print("\n📁 Generated Files:")
    print(f"  Full frames:     {full_dir}/ ({len(list(full_dir.glob('*.png')))} files)")
    print(f"  Cropped frames:  {cropped_dir}/ ({len(list(cropped_dir.glob('*.png')))} files)")
    
    if os.path.exists("full_results.csv"):
        print(f"  Full results:    full_results.csv ✅")
    else:
        print(f"  Full results:    full_results.csv ❌")
        
    if os.path.exists("cropped_results.csv"):
        print(f"  Cropped results: cropped_results.csv ✅")
    else:
        print(f"  Cropped results: cropped_results.csv ❌")
    
    if viewers:
        print("\n⚠️  Press Ctrl+C to stop the viewers")
    
    if viewers:
        try:
            # Keep script running while viewers are active
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping viewers...")
            for name, url, process in viewers:
                if process:
                    process.terminate()
                    print(f"✓ Stopped {name} viewer")
            print("✓ All viewers stopped")
    else:
        print("No viewers to manage.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 