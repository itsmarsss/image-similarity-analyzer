"""
helper_select_and_crop.py

Multi-crop tool for extracting image pairs from video sequences. Loads two videos 
side-by-side, allows navigation through frames, and enables selection of multiple 
regions of interest (ROI) for analysis. Automatically generates CSV batch files 
for use with the main analysis tool.

Features:
- Side-by-side video comparison
- Frame navigation with trackbar and keyboard shortcuts
- Multiple ROI selection per session
- Numbered output files (prefix_input_001.png, prefix_output_001.png, etc.)
- Automatic CSV batch file generation
- Session summary with extraction statistics

Usage:
    python helper_select_and_crop.py --input input.mp4 --output output.mp4 --prefix body_swap

Options:
    --input,  -i   Path to original input video
    --output, -o   Path to processed/swapped output video
    --width,  -w   Width of display window (default 1200)
    --prefix, -p   Filename prefix for output files (default 'pair')

Controls:
    - Trackbar: Navigate through frames
    - 's' key: Select ROI and save current pair
    - 'n' key: Next frame (advance by 1)
    - 'b' key: Previous frame (go back by 1)
    - 'r' key: Reset to frame 0
    - 'q' key: Quit and show session summary

Output:
    - Individual image pairs: prefix_input_XXX.png, prefix_output_XXX.png
    - CSV batch file: prefix_batch_list.csv (ready for main.py --batch)
"""
import cv2
import argparse
import numpy as np
import os
import csv


def get_frame(cap, idx):
    """Retrieve frame at index idx."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Could not get frame {idx}")
    return frame


def on_trackbar(pos):
    global frame_idx
    frame_idx = pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',  required=True, help='Path to input.mp4')
    parser.add_argument('-o', '--output', required=True, help='Path to output.mp4')
    parser.add_argument('-w', '--width',  type=int, default=1200, help='Window width')
    parser.add_argument('-p', '--prefix', default='pair', help='Filename prefix (default: pair)')
    args = parser.parse_args()

    cap_in  = cv2.VideoCapture(args.input)
    cap_out = cv2.VideoCapture(args.output)
    total_frames = int(min(cap_in.get(cv2.CAP_PROP_FRAME_COUNT),
                           cap_out.get(cv2.CAP_PROP_FRAME_COUNT)))

    if total_frames == 0:
        print("No frames found in one of the videos.")
        return

    window_name = 'Frame Selector - Multi-Crop Tool'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, 600)

    global frame_idx
    frame_idx = 0
    cv2.createTrackbar('Frame', window_name, 0, total_frames - 1, on_trackbar)

    bbox = None
    crop_counter = 1
    saved_pairs = []

    print(f"=== Multi-Crop Tool ===")
    print(f"Total frames: {total_frames}")
    print(f"Controls:")
    print(f"  's' - Select ROI and save pair")
    print(f"  'n' - Next frame")
    print(f"  'b' - Previous frame") 
    print(f"  'r' - Reset to frame 0")
    print(f"  'q' - Quit")
    print(f"Filename format: {args.prefix}_input_XXX.png, {args.prefix}_output_XXX.png")
    print("=" * 40)

    while True:
        # Fetch frames
        try:
            frame1 = get_frame(cap_in, frame_idx)
            frame2 = get_frame(cap_out, frame_idx)
        except ValueError as e:
            print(f"Error: {e}")
            frame_idx = max(0, frame_idx - 1)
            cv2.setTrackbarPos('Frame', window_name, frame_idx)
            continue

        # Resize to same height for side-by-side
        h = min(frame1.shape[0], frame2.shape[0])
        frame1 = cv2.resize(frame1, (int(frame1.shape[1] * h / frame1.shape[0]), h))
        frame2 = cv2.resize(frame2, (int(frame2.shape[1] * h / frame2.shape[0]), h))

        combined = np.hstack([frame1, frame2])
        disp = combined.copy()

        # Draw current ROI if selected
        if bbox is not None:
            x, y, w, h_box = bbox
            cv2.rectangle(disp, (x, y), (x+w, y+h_box), (0,255,0), 2)
            cv2.putText(disp, f"ROI Selected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Add info overlay
        info_text = [
            f"Frame: {frame_idx}/{total_frames-1}",
            f"Saved pairs: {len(saved_pairs)}",
            f"Next pair: {args.prefix}_input_{crop_counter:03d}.png"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(disp, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(disp, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        cv2.imshow(window_name, disp)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('s'):
            # Select ROI on combined image
            print(f"Select ROI for pair {crop_counter:03d}...")
            bbox = cv2.selectROI(window_name, combined, False)
            
            if bbox[2] > 0 and bbox[3] > 0:  # Valid selection
                x, y, w, h_box = map(int, bbox)
                
                # Determine which frame the ROI belongs to
                w1 = frame1.shape[1]
                
                if x + w <= w1:
                    # ROI is entirely in first frame
                    crop1 = frame1[y:y+h_box, x:x+w]
                    crop2 = frame2[y:y+h_box, x:x+w]  # Same coordinates on second frame
                elif x >= w1:
                    # ROI is entirely in second frame
                    x_adj = x - w1
                    crop1 = frame1[y:y+h_box, x_adj:x_adj+w]
                    crop2 = frame2[y:y+h_box, x_adj:x_adj+w]
                else:
                    # ROI spans both frames - use the part that's more in frame 1 or 2
                    if x + w/2 <= w1:
                        # More in frame 1
                        crop1 = frame1[y:y+h_box, x:min(x+w, w1)]
                        crop2 = frame2[y:y+h_box, x:min(x+w, w1)]
                    else:
                        # More in frame 2
                        x_adj = max(0, x - w1)
                        crop1 = frame1[y:y+h_box, x_adj:x_adj+w]
                        crop2 = frame2[y:y+h_box, x_adj:x_adj+w]
                
                # Save crops with numbered filenames
                input_filename = f"{args.prefix}_input_{crop_counter:03d}.png"
                output_filename = f"{args.prefix}_output_{crop_counter:03d}.png"
                
                cv2.imwrite(input_filename, crop1)
                cv2.imwrite(output_filename, crop2)
                
                pair_info = {
                    'frame': frame_idx,
                    'input_file': input_filename,
                    'output_file': output_filename,
                    'bbox': bbox
                }
                saved_pairs.append(pair_info)
                
                print(f"✓ Saved pair {crop_counter:03d}: {input_filename}, {output_filename}")
                print(f"  Frame: {frame_idx}, ROI: {bbox}")
                
                crop_counter += 1
                bbox = None  # Reset selection
            else:
                print("Invalid ROI selection. Try again.")
                
        elif key == ord('n'):
            # Next frame
            frame_idx = min(frame_idx + 1, total_frames - 1)
            cv2.setTrackbarPos('Frame', window_name, frame_idx)
            
        elif key == ord('b'):
            # Previous frame
            frame_idx = max(frame_idx - 1, 0)
            cv2.setTrackbarPos('Frame', window_name, frame_idx)
            
        elif key == ord('r'):
            # Reset to frame 0
            frame_idx = 0
            cv2.setTrackbarPos('Frame', window_name, frame_idx)
            
        elif key == ord('q'):
            break

    # Summary
    print("\n" + "=" * 50)
    print(f"SESSION SUMMARY")
    print(f"Total pairs saved: {len(saved_pairs)}")
    
    if saved_pairs:
        print(f"Saved files:")
        for pair in saved_pairs:
            print(f"  Frame {pair['frame']:03d}: {pair['input_file']}, {pair['output_file']}")
            
        # Create a batch CSV file for easy processing
        batch_filename = f"{args.prefix}_batch_list.csv"
        with open(batch_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['input_path', 'output_path'])  # CSV header
            for pair in saved_pairs:
                writer.writerow([pair['input_file'], pair['output_file']])
        print(f"\nBatch list saved: {batch_filename}")
        print(f"Use this with: python main.py --batch {batch_filename} --cohere-key YOUR_KEY")
    
    print("=" * 50)

    cap_in.release()
    cap_out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
