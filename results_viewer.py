"""
results_viewer.py

Interactive visualization tool for image similarity analysis results.
Displays image pairs side by side with score breakdowns and allows navigation
between pairs using a slider.

Usage:
    python results_viewer.py --results similarity_results_20231201_143000.csv
    python results_viewer.py --results similarity_results_20231201_143000.txt
    python results_viewer.py --results pair_batch_list.txt --format batch

Options:
    --results, -r   Path to results file (CSV, TXT, or batch list)
    --format, -f    File format: 'csv', 'txt', or 'batch' (auto-detected if not specified)
    --width, -w     Window width (default: 1400)
    --height        Window height (default: 800)
"""

import cv2
import argparse
import numpy as np
import csv
import os
from typing import List, Dict, Optional


def load_csv_results(filepath: str) -> List[Dict]:
    """Load results from CSV file."""
    results = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['success'].lower() == 'true':
                    results.append({
                        'input_path': row['input_path'],
                        'output_path': row['output_path'],
                        'pixel_score': float(row['pixel_score']),
                        'embedding_score': float(row['embedding_score']),
                        'pose_score': float(row['pose_score']),
                        'combined_score': float(row['combined_score']),
                        'success': True
                    })
                else:
                    results.append({
                        'input_path': row['input_path'],
                        'output_path': row['output_path'],
                        'error': row.get('error', 'Unknown error'),
                        'success': False
                    })
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return []
    return results


def load_batch_file(filepath: str) -> List[Dict]:
    """Load pairs from batch file (for when results aren't computed yet)."""
    results = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split(',')
                if len(parts) != 2:
                    continue
                    
                input_path, output_path = parts[0].strip(), parts[1].strip()
                results.append({
                    'input_path': input_path,
                    'output_path': output_path,
                    'pixel_score': None,
                    'embedding_score': None,
                    'pose_score': None,
                    'combined_score': None,
                    'success': None  # Indicates not computed
                })
    except Exception as e:
        print(f"Error loading batch file: {e}")
        return []
    return results


def load_txt_results(filepath: str) -> List[Dict]:
    """Load results from text file."""
    results = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        current_result = None
        for line in lines:
            line = line.strip()
            if line.startswith('Pair ') and ':' in line:
                # Parse pair line: "Pair 001: input.png -> output.png"
                if current_result:
                    results.append(current_result)
                
                pair_info = line.split(':', 1)[1].strip()
                if '->' in pair_info:
                    input_path, output_path = pair_info.split('->', 1)
                    current_result = {
                        'input_path': input_path.strip(),
                        'output_path': output_path.strip(),
                        'success': True
                    }
            elif current_result and line.startswith('Pixel Score:'):
                current_result['pixel_score'] = float(line.split(':')[1].strip())
            elif current_result and line.startswith('Embedding Score:'):
                current_result['embedding_score'] = float(line.split(':')[1].strip())
            elif current_result and line.startswith('Pose Score:'):
                current_result['pose_score'] = float(line.split(':')[1].strip())
            elif current_result and line.startswith('Combined Score:'):
                current_result['combined_score'] = float(line.split(':')[1].strip())
            elif current_result and line.startswith('ERROR:'):
                current_result['error'] = line.split(':', 1)[1].strip()
                current_result['success'] = False
        
        if current_result:
            results.append(current_result)
            
    except Exception as e:
        print(f"Error loading text file: {e}")
        return []
    return results


def detect_format(filepath: str) -> str:
    """Auto-detect file format based on extension and content."""
    if filepath.endswith('.csv'):
        return 'csv'
    elif filepath.endswith('.txt'):
        # Check if it's a batch file or results file
        try:
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                if 'Image Similarity Analysis Results' in first_line:
                    return 'txt'
                else:
                    return 'batch'
        except:
            return 'txt'
    else:
        return 'batch'


def load_and_resize_image(path: str, target_height: int) -> Optional[np.ndarray]:
    """Load and resize image to target height while maintaining aspect ratio."""
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        
        h, w = img.shape[:2]
        new_width = int(w * target_height / h)
        resized = cv2.resize(img, (new_width, target_height))
        return resized
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def create_score_overlay(scores: Dict, width: int, height: int) -> np.ndarray:
    """Create an overlay image with score information."""
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    overlay.fill(40)  # Dark background
    
    if scores['success'] is None:
        # Not computed yet
        texts = [
            "Scores not computed",
            "Run analysis first:",
            "python main.py --batch <file>",
            "--cohere-key <key>"
        ]
        y_start = 40
        line_spacing = 25
    elif scores['success']:
        # Successful analysis
        texts = [
            f"Pixel Score:     {scores['pixel_score']:.4f}",
            f"Embedding Score: {scores['embedding_score']:.4f}",
            f"Pose Score:      {scores['pose_score']:.4f}",
            f"Combined Score:  {scores['combined_score']:.4f}",
            "",
            "Interpretation:",
            f"  0.0-0.2: Very similar",
            f"  0.2-0.4: Moderately similar", 
            f"  0.4-0.6: Somewhat different",
            f"  0.6-0.8: Very different",
            f"  0.8-1.0: Extremely different"
        ]
        y_start = 25
        line_spacing = 22
    else:
        # Error
        texts = [
            "Analysis failed:",
            f"Error: {scores.get('error', 'Unknown error')[:40]}..."
        ]
        y_start = 40
        line_spacing = 25
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45  # Smaller font
    thickness = 1      # Thinner for sharper text
    
    for i, text in enumerate(texts):
        y = y_start + i * line_spacing
        if y < height - 15:
            # Add subtle shadow for better readability
            cv2.putText(overlay, text, (11, y+1), font, font_scale, (0, 0, 0), thickness)
            cv2.putText(overlay, text, (10, y), font, font_scale, (255, 255, 255), thickness)
    
    return overlay


def on_mouse(event, x, y, flags, param):
    global current_pair, total_pairs, image_area_width
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Click navigation: left side = previous, right side = next
        if x < image_area_width // 3:
            # Left third - go to previous
            current_pair = max(current_pair - 1, 0)
        elif x > 2 * image_area_width // 3:
            # Right third - go to next  
            current_pair = min(current_pair + 1, total_pairs - 1)


def main():
    parser = argparse.ArgumentParser(description='Interactive Image Similarity Results Viewer')
    parser.add_argument('-r', '--results', required=True, help='Path to results file')
    parser.add_argument('-f', '--format', choices=['csv', 'txt', 'batch'], 
                       help='File format (auto-detected if not specified)')
    parser.add_argument('-w', '--width', type=int, default=1400, help='Window width')
    parser.add_argument('--height', type=int, default=800, help='Window height')
    args = parser.parse_args()
    
    # Detect format if not specified
    file_format = args.format or detect_format(args.results)
    
    # Load results
    print(f"Loading results from {args.results} (format: {file_format})")
    if file_format == 'csv':
        results = load_csv_results(args.results)
    elif file_format == 'txt':
        results = load_txt_results(args.results)
    elif file_format == 'batch':
        results = load_batch_file(args.results)
    else:
        print(f"Unknown format: {file_format}")
        return
    
    if not results:
        print("No results found or failed to load file!")
        return
    
    print(f"Loaded {len(results)} pairs")
    
    # Setup window
    window_name = 'Image Similarity Results Viewer'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)
    
    global current_pair, total_pairs, image_area_width
    current_pair = 0
    total_pairs = len(results)
    
    # Calculate layout
    score_panel_width = 350
    image_area_width = args.width - score_panel_width
    target_img_height = args.height - 80  # Leave space for info
    
    # Set mouse callback for click navigation
    cv2.setMouseCallback(window_name, on_mouse)
    
    print(f"\nNavigation Controls:")
    print(f"  Left/Right arrows: Previous/Next pair")
    print(f"  Up/Down arrows: Jump +/-5 pairs")
    print(f"  Click left/right sides: Previous/Next pair")
    print(f"  Space/Backspace: Next/Previous pair")
    print(f"  Number keys (1-9,0): Jump to specific pair")
    print(f"  Home/End: First/Last pair")
    print(f"  'q' or Escape: Quit")
    print(f"  's': Print current pair info")
    print("=" * 50)
    
    while True:
        result = results[current_pair]
        
        # Load images
        img1 = load_and_resize_image(result['input_path'], target_img_height)
        img2 = load_and_resize_image(result['output_path'], target_img_height)
        
        if img1 is None:
            img1 = np.zeros((target_img_height, 300, 3), dtype=np.uint8)
            cv2.putText(img1, "Image not found", (10, target_img_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if img2 is None:
            img2 = np.zeros((target_img_height, 300, 3), dtype=np.uint8)
            cv2.putText(img2, "Image not found", (10, target_img_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Ensure images have same height
        h = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
        img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))
        
        # Create side-by-side comparison
        combined_images = np.hstack([img1, img2])
        
        # Scale to fit available width
        if combined_images.shape[1] > image_area_width:
            scale = image_area_width / combined_images.shape[1]
            new_width = int(combined_images.shape[1] * scale)
            new_height = int(combined_images.shape[0] * scale)
            combined_images = cv2.resize(combined_images, (new_width, new_height))
        
        # Add subtle click zone indicators
        img_height, img_width = combined_images.shape[:2]
        
        # Left click zone (previous) - only if not first pair
        if current_pair > 0:
            left_zone_width = img_width // 3
            overlay = combined_images.copy()
            cv2.rectangle(overlay, (0, 0), (left_zone_width, img_height), (100, 100, 255), -1)
            combined_images = cv2.addWeighted(combined_images, 0.95, overlay, 0.05, 0)
            
            # Add "PREV" text
            cv2.putText(combined_images, "PREV", (10, img_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
        
        # Right click zone (next) - only if not last pair  
        if current_pair < len(results) - 1:
            right_zone_start = 2 * img_width // 3
            overlay = combined_images.copy()
            cv2.rectangle(overlay, (right_zone_start, 0), (img_width, img_height), (100, 255, 100), -1)
            combined_images = cv2.addWeighted(combined_images, 0.95, overlay, 0.05, 0)
            
            # Add "NEXT" text
            cv2.putText(combined_images, "NEXT", (img_width - 70, img_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)
        
        # Create score panel
        score_panel = create_score_overlay(result, score_panel_width, combined_images.shape[0])
        
        # Create score panel to match image height
        score_panel = create_score_overlay(result, score_panel_width, combined_images.shape[0])
        
        # Combine images and score panel horizontally
        main_content = np.hstack([combined_images, score_panel])
        
        # Create header at the top
        header_height = 70
        header = np.zeros((header_height, main_content.shape[1], 3), dtype=np.uint8)
        header.fill(30)
        
        # Add header text
        pair_text = f"Pair {current_pair + 1}/{len(results)}"
        input_name = os.path.basename(result['input_path'])
        output_name = os.path.basename(result['output_path'])
        files_text = f"Input: {input_name} | Output: {output_name}"
        
        # Main title with shadow
        cv2.putText(header, pair_text, (11, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(header, pair_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # File names with shadow  
        cv2.putText(header, files_text, (11, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(header, files_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Navigation hints on the right side of header
        nav_text = "< > Arrows | Click sides"
        text_size = cv2.getTextSize(nav_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        nav_x = header.shape[1] - text_size[0] - 10
        cv2.putText(header, nav_text, (nav_x + 1, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(header, nav_text, (nav_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # Stack header on top of main content
        final_display = np.vstack([header, main_content])
        
        cv2.imshow(window_name, final_display)
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or Escape
            break
        elif key == 81 or key == 2:  # Left arrow key  
            current_pair = max(current_pair - 1, 0)
        elif key == 83 or key == 3:  # Right arrow key
            current_pair = min(current_pair + 1, len(results) - 1)
        elif key == 82:  # Up arrow - jump back 5
            current_pair = max(current_pair - 5, 0)
        elif key == 84:  # Down arrow - jump forward 5
            current_pair = min(current_pair + 5, len(results) - 1)
        elif key == ord(' '):  # Spacebar - next pair
            current_pair = min(current_pair + 1, len(results) - 1)
        elif key == 8:  # Backspace - previous pair
            current_pair = max(current_pair - 1, 0)
        elif key >= ord('1') and key <= ord('9'):  # Number keys 1-9
            target = int(chr(key)) - 1
            if target < len(results):
                current_pair = target
        elif key == ord('0'):  # 0 key - go to pair 10 (if exists)
            if len(results) > 9:
                current_pair = 9
        elif key == 71:  # Home key - first pair
            current_pair = 0
        elif key == 79:  # End key - last pair
            current_pair = len(results) - 1
        elif key == ord('s'):
            print(f"\nPair {current_pair + 1} Info:")
            print(f"  Input:  {result['input_path']}")
            print(f"  Output: {result['output_path']}")
            if result['success']:
                print(f"  Pixel Score:     {result['pixel_score']:.4f}")
                print(f"  Embedding Score: {result['embedding_score']:.4f}")
                print(f"  Pose Score:      {result['pose_score']:.4f}")
                print(f"  Combined Score:  {result['combined_score']:.4f}")
            elif result['success'] is False:
                print(f"  Error: {result.get('error', 'Unknown error')}")
            else:
                print(f"  Status: Not analyzed yet")
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 