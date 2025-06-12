"""
helper_select_and_crop.py

This script loads two videos (input.mp4 and output.mp4), displays them side by side,
provides a trackbar to select a frame index, allows the user to draw a single ROI on
the combined view (applied to both frames), and saves the cropped regions as input.png and output.png.

Usage:
    python helper_select_and_crop.py --input input.mp4 --output output.mp4 --width 1280

Options:
    --input,  -i   Path to original input video
    --output, -o   Path to self-swap output video
    --width,  -w   Width of display window (default 1200)
"""
import cv2
import argparse
import numpy as np


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
    args = parser.parse_args()

    cap_in  = cv2.VideoCapture(args.input)
    cap_out = cv2.VideoCapture(args.output)
    total_frames = int(min(cap_in.get(cv2.CAP_PROP_FRAME_COUNT),
                           cap_out.get(cv2.CAP_PROP_FRAME_COUNT)))

    if total_frames == 0:
        print("No frames found in one of the videos.")
        return

    window_name = 'Frame Selector'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, 600)

    global frame_idx
    frame_idx = 0
    cv2.createTrackbar('Frame', window_name, 0, total_frames - 1, on_trackbar)

    bbox = None

    while True:
        # Fetch frames
        frame1 = get_frame(cap_in, frame_idx)
        frame2 = get_frame(cap_out, frame_idx)

        # Resize to same height for side-by-side
        h = min(frame1.shape[0], frame2.shape[0])
        frame1 = cv2.resize(frame1, (int(frame1.shape[1] * h / frame1.shape[0]), h))
        frame2 = cv2.resize(frame2, (int(frame2.shape[1] * h / frame2.shape[0]), h))

        combined = np.hstack([frame1, frame2])
        disp = combined.copy()

        if bbox is not None:
            x, y, w, h_box = bbox
            cv2.rectangle(disp, (x, y), (x+w, y+h_box), (0,255,0), 2)

        cv2.imshow(window_name, disp)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('s'):
            # select ROI on combined image
            bbox = cv2.selectROI(window_name, combined, False)
            # crop and save
            x, y, w, h_box = map(int, bbox)
            # compute crop relative to frame1 and frame2
            w1 = frame1.shape[1]
            if x < w1:
                box1 = (x, y, w, h_box)
                box2 = (x - w1 if x >= w1 else 0, y, w, h_box)
            # Actually easier: apply same pixel coords on each original resized
            # Save crops
            crop1 = frame1[y:y+h_box, x:x+w]
            crop2 = frame2[y:y+h_box, x:x+w]
            cv2.imwrite('input.png', crop1)
            cv2.imwrite('output.png', crop2)
            print(f"Saved input.png and output.png from frame {frame_idx} and bbox {bbox}")
            break
        elif key == ord('q'):
            print("Exiting without saving.")
            break

    cap_in.release()
    cap_out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
