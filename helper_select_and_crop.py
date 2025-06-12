"""
gradio_helper_select_and_crop.py

Web-based multi-crop tool for extracting image pairs from video sequences using Gradio.
Provides an intuitive web interface for loading two videos side-by-side, navigating through 
frames, and selecting multiple regions of interest (ROI) for analysis.

Features:
- Web-based interface (no OpenCV window needed)
- Side-by-side video frame comparison
- Frame navigation with slider
- Interactive crop region selection via coordinates
- Multiple ROI selection per session
- Automatic CSV batch file generation
- Session download with all crops and batch file

Usage:
    python gradio_helper_select_and_crop.py
    
Then open the provided URL in your browser and upload your videos.
"""

import gradio as gr
import cv2
import numpy as np
import os
import csv
import tempfile
import zipfile
from PIL import Image
import io
import base64


class VideoCropTool:
    def __init__(self):
        self.cap_input = None
        self.cap_output = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.crop_counter = 1
        self.saved_pairs = []
        self.temp_dir = tempfile.mkdtemp()
        self.prefix = "pair"
        
    def load_videos(self, input_video, output_video, prefix):
        """Load the input and output videos"""
        if input_video is None or output_video is None:
            return "Please upload both input and output videos.", None, None, 0, []
            
        self.prefix = prefix if prefix else "pair"
        
        # Release previous videos if any
        if self.cap_input:
            self.cap_input.release()
        if self.cap_output:
            self.cap_output.release()
            
        self.cap_input = cv2.VideoCapture(input_video)
        self.cap_output = cv2.VideoCapture(output_video)
        
        total_input = int(self.cap_input.get(cv2.CAP_PROP_FRAME_COUNT))
        total_output = int(self.cap_output.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = min(total_input, total_output)
        
        if self.total_frames == 0:
            return "Error: No frames found in one or both videos.", None, None, 0, []
            
        self.current_frame_idx = 0
        self.crop_counter = 1
        self.saved_pairs = []
        
        # Get first frame
        combined_frame, frame_info = self.get_combined_frame(0)
        
        return f"Videos loaded successfully! Total frames: {self.total_frames}", combined_frame, frame_info, self.total_frames - 1, []
    
    def get_frame(self, cap, idx):
        """Retrieve frame at specific index"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame
    
    def get_combined_frame(self, frame_idx):
        """Get combined side-by-side frame"""
        if not self.cap_input or not self.cap_output:
            return None, "No videos loaded"
            
        frame1 = self.get_frame(self.cap_input, frame_idx)
        frame2 = self.get_frame(self.cap_output, frame_idx)
        
        if frame1 is None or frame2 is None:
            return None, f"Could not get frame {frame_idx}"
            
        # Resize to same height for side-by-side display
        h = min(frame1.shape[0], frame2.shape[0])
        w1 = int(frame1.shape[1] * h / frame1.shape[0])
        w2 = int(frame2.shape[1] * h / frame2.shape[0])
        
        frame1_resized = cv2.resize(frame1, (w1, h))
        frame2_resized = cv2.resize(frame2, (w2, h))
        
        # Combine frames side by side
        combined = np.hstack([frame1_resized, frame2_resized])
        
        # Convert BGR to RGB for display
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        
        # Store original frames and dimensions for cropping
        self.current_frames = {
            'input': frame1,
            'output': frame2,
            'input_resized': frame1_resized,
            'output_resized': frame2_resized,
            'combined_width_split': w1
        }
        
        frame_info = f"Frame {frame_idx}/{self.total_frames-1} | Saved pairs: {len(self.saved_pairs)} | Next: {self.prefix}_{self.crop_counter:03d}"
        
        return Image.fromarray(combined_rgb), frame_info
    
    def navigate_frame(self, frame_idx):
        """Navigate to specific frame"""
        self.current_frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        return self.get_combined_frame(self.current_frame_idx)
    
    def crop_and_save(self, x1, y1, x2, y2):
        """Crop and save the selected region"""
        if not hasattr(self, 'current_frames'):
            return "No frame loaded. Please navigate to a frame first.", None
            
        if x1 >= x2 or y1 >= y2:
            return "Invalid crop coordinates. Make sure x2 > x1 and y2 > y1.", None
            
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get current frames
        frames = self.current_frames
        w1 = frames['combined_width_split']
        
        # Determine which frame(s) the crop region belongs to
        input_crop = None
        output_crop = None
        
        # Scale coordinates back to original frame size
        input_frame = frames['input']
        output_frame = frames['output']
        input_resized = frames['input_resized']
        output_resized = frames['output_resized']
        
        # Scale factors
        input_scale_x = input_frame.shape[1] / input_resized.shape[1]
        input_scale_y = input_frame.shape[0] / input_resized.shape[0]
        output_scale_x = output_frame.shape[1] / output_resized.shape[1]
        output_scale_y = output_frame.shape[0] / output_resized.shape[0]
        
        if x2 <= w1:
            # Crop is entirely in input frame (left side)
            orig_x1 = int(x1 * input_scale_x)
            orig_y1 = int(y1 * input_scale_y)
            orig_x2 = int(x2 * input_scale_x)
            orig_y2 = int(y2 * input_scale_y)
            
            input_crop = input_frame[orig_y1:orig_y2, orig_x1:orig_x2]
            output_crop = output_frame[orig_y1:orig_y2, orig_x1:orig_x2]
            
        elif x1 >= w1:
            # Crop is entirely in output frame (right side)
            adj_x1 = x1 - w1
            adj_x2 = x2 - w1
            
            orig_x1 = int(adj_x1 * output_scale_x)
            orig_y1 = int(y1 * output_scale_y)
            orig_x2 = int(adj_x2 * output_scale_x)
            orig_y2 = int(y2 * output_scale_y)
            
            input_crop = input_frame[orig_y1:orig_y2, orig_x1:orig_x2]
            output_crop = output_frame[orig_y1:orig_y2, orig_x1:orig_x2]
            
        else:
            # Crop spans both frames - choose the side with more area
            left_area = w1 - x1
            right_area = x2 - w1
            
            if left_area >= right_area:
                # Use input frame coordinates
                orig_x1 = int(x1 * input_scale_x)
                orig_y1 = int(y1 * input_scale_y)
                orig_x2 = int(min(x2, w1) * input_scale_x)
                orig_y2 = int(y2 * input_scale_y)
                
                input_crop = input_frame[orig_y1:orig_y2, orig_x1:orig_x2]
                output_crop = output_frame[orig_y1:orig_y2, orig_x1:orig_x2]
            else:
                # Use output frame coordinates
                adj_x1 = max(x1 - w1, 0)
                adj_x2 = x2 - w1
                
                orig_x1 = int(adj_x1 * output_scale_x)
                orig_y1 = int(y1 * output_scale_y)
                orig_x2 = int(adj_x2 * output_scale_x)
                orig_y2 = int(y2 * output_scale_y)
                
                input_crop = input_frame[orig_y1:orig_y2, orig_x1:orig_x2]
                output_crop = output_frame[orig_y1:orig_y2, orig_x1:orig_x2]
        
        if input_crop is None or output_crop is None or input_crop.size == 0 or output_crop.size == 0:
            return "Failed to extract crop region. Please check coordinates.", None
            
        # Save crops
        input_filename = f"{self.prefix}_input_{self.crop_counter:03d}.png"
        output_filename = f"{self.prefix}_output_{self.crop_counter:03d}.png"
        
        input_path = os.path.join(self.temp_dir, input_filename)
        output_path = os.path.join(self.temp_dir, output_filename)
        
        cv2.imwrite(input_path, input_crop)
        cv2.imwrite(output_path, output_crop)
        
        # Store pair info
        pair_info = {
            'frame': self.current_frame_idx,
            'input_file': input_filename,
            'output_file': output_filename,
            'crop_coords': (x1, y1, x2, y2)
        }
        self.saved_pairs.append(pair_info)
        
        result_msg = f"✓ Saved pair {self.crop_counter:03d}: {input_filename}, {output_filename}\n"
        result_msg += f"Frame: {self.current_frame_idx}, Coords: ({x1}, {y1}, {x2}, {y2})"
        
        self.crop_counter += 1
        
        return result_msg, self.get_session_summary()
    
    def get_session_summary(self):
        """Get current session summary"""
        if not self.saved_pairs:
            return "No pairs saved yet."
            
        summary = f"Session Summary:\n"
        summary += f"Total pairs saved: {len(self.saved_pairs)}\n\n"
        
        for pair in self.saved_pairs:
            summary += f"Frame {pair['frame']:03d}: {pair['input_file']}, {pair['output_file']}\n"
            
        return summary
    
    def create_download_package(self):
        """Create a ZIP file with all saved pairs and batch CSV"""
        if not self.saved_pairs:
            return None, "No pairs to download."
            
        # Create batch CSV
        batch_filename = f"{self.prefix}_batch_list.csv"
        batch_path = os.path.join(self.temp_dir, batch_filename)
        
        with open(batch_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['input_path', 'output_path'])
            for pair in self.saved_pairs:
                writer.writerow([pair['input_file'], pair['output_file']])
        
        # Create ZIP file
        zip_filename = f"{self.prefix}_crops_package.zip"
        zip_path = os.path.join(self.temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add batch CSV
            zipf.write(batch_path, batch_filename)
            
            # Add all image pairs
            for pair in self.saved_pairs:
                input_path = os.path.join(self.temp_dir, pair['input_file'])
                output_path = os.path.join(self.temp_dir, pair['output_file'])
                
                zipf.write(input_path, pair['input_file'])
                zipf.write(output_path, pair['output_file'])
        
        return zip_path, f"Download package created with {len(self.saved_pairs)} pairs and batch CSV file."


def create_interface():
    tool = VideoCropTool()
    
    def load_videos_wrapper(input_video, output_video, prefix):
        return tool.load_videos(input_video, output_video, prefix)
    
    def navigate_wrapper(frame_idx):
        return tool.navigate_frame(frame_idx)
    
    def crop_wrapper(x1, y1, x2, y2):
        return tool.crop_and_save(x1, y1, x2, y2)
    
    def download_wrapper():
        return tool.create_download_package()
    
    with gr.Blocks(title="Video Crop Tool", theme=gr.themes.Soft()) as iface:
        gr.Markdown("# 🎬 Video Multi-Crop Tool")
        gr.Markdown("Upload two videos, navigate through frames, and select regions to crop for image similarity analysis.")
        
        with gr.Row():
            with gr.Column():
                input_video = gr.File(label="📥 Input Video", file_types=[".mp4", ".avi", ".mov", ".mkv"])
                output_video = gr.File(label="📤 Output Video", file_types=[".mp4", ".avi", ".mov", ".mkv"])
                prefix_input = gr.Textbox(label="🏷️ Filename Prefix", value="pair", placeholder="Enter prefix for output files")
                load_btn = gr.Button("🚀 Load Videos", variant="primary")
        
        status_text = gr.Textbox(label="📊 Status", interactive=False)
        
        with gr.Row():
            with gr.Column():
                frame_display = gr.Image(label="🖼️ Video Frames (Input | Output)", type="pil")
                frame_info = gr.Textbox(label="📝 Frame Info", interactive=False)
                
                with gr.Row():
                    frame_slider = gr.Slider(
                        minimum=0, maximum=100, step=1, value=0,
                        label="🎯 Frame Navigation",
                        interactive=True
                    )
                
            with gr.Column():
                gr.Markdown("### 🎯 Crop Selection")
                gr.Markdown("Enter coordinates to define crop region:")
                
                with gr.Row():
                    x1_input = gr.Number(label="X1 (left)", value=0, precision=0)
                    y1_input = gr.Number(label="Y1 (top)", value=0, precision=0)
                
                with gr.Row():
                    x2_input = gr.Number(label="X2 (right)", value=100, precision=0)
                    y2_input = gr.Number(label="Y2 (bottom)", value=100, precision=0)
                
                crop_btn = gr.Button("✂️ Crop & Save", variant="secondary")
                crop_result = gr.Textbox(label="✅ Crop Result", interactive=False)
        
        with gr.Row():
            with gr.Column():
                session_summary = gr.Textbox(label="📋 Session Summary", interactive=False, lines=8)
            
            with gr.Column():
                download_btn = gr.Button("📦 Create Download Package", variant="primary")
                download_file = gr.File(label="⬇️ Download Package")
                download_status = gr.Textbox(label="📦 Package Status", interactive=False)
        
        # Event handlers
        load_btn.click(
            load_videos_wrapper,
            inputs=[input_video, output_video, prefix_input],
            outputs=[status_text, frame_display, frame_info, frame_slider, session_summary]
        )
        
        frame_slider.change(
            navigate_wrapper,
            inputs=[frame_slider],
            outputs=[frame_display, frame_info]
        )
        
        crop_btn.click(
            crop_wrapper,
            inputs=[x1_input, y1_input, x2_input, y2_input],
            outputs=[crop_result, session_summary]
        )
        
        download_btn.click(
            download_wrapper,
            outputs=[download_file, download_status]
        )
        
        # Instructions
        with gr.Accordion("📖 Instructions", open=False):
            gr.Markdown("""
            ### How to use:
            1. **Upload Videos**: Select your input and output video files
            2. **Set Prefix**: Choose a prefix for your output files (default: 'pair')
            3. **Load Videos**: Click 'Load Videos' to start
            4. **Navigate**: Use the slider to move through frames
            5. **Select Crop Region**: Enter coordinates (X1,Y1) for top-left and (X2,Y2) for bottom-right
            6. **Save Crops**: Click 'Crop & Save' to extract and save the region
            7. **Repeat**: Navigate to different frames and select more regions as needed
            8. **Download**: Click 'Create Download Package' to get all crops and batch CSV file
            
            ### Coordinate System:
            - The display shows both videos side-by-side
            - X coordinates: 0 to combined width (left video | right video)
            - Y coordinates: 0 to height (top to bottom)
            - Crops can be from either video or spanning both
            
            ### Output:
            - Individual image pairs: `prefix_input_001.png`, `prefix_output_001.png`, etc.
            - Batch CSV file: `prefix_batch_list.csv` (ready for main.py --batch)
            """)
    
    return iface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    ) 