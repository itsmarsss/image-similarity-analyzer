"""
gradio_helper_select_and_crop.py

Web-based multi-crop tool for extracting image pairs from video sequences using Gradio.
Provides an intuitive web interface for loading two videos side-by-side, navigating through 
frames, and selecting multiple regions of interest (ROI) for analysis.

Features:
- Web-based interface (no OpenCV window needed)
- Side-by-side video frame comparison
- Frame navigation with slider
- Interactive drag-and-drop crop region selection
- Coordinate-based manual selection (backup)
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
from PIL import Image, ImageDraw
import io
import base64
import json


class VideoCropTool:
    def __init__(self):
        self.cap_input = None
        self.cap_output = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.crop_counter = 0
        self.saved_pairs = []
        self.temp_dir = tempfile.mkdtemp()
        self.prefix = "pair"
        self.current_selection = None
        self.roi_coordinates = {
            'x_temp': 0,
            'y_temp': 0,
            'x_new': 0,
            'y_new': 0,
            'clicks': 0,
        }
        
    def load_videos(self, input_video, output_video, prefix):
        """Load the input and output videos"""
        if input_video is None or output_video is None:
            return "Please upload both input and output videos.", None, None, None, None, 0, [], ""
            
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
            return "Error: No frames found in one or both videos.", None, None, None, None, 0, [], ""
            
        self.current_frame_idx = 0
        self.crop_counter = 0
        self.saved_pairs = []
        # Reset ROI coordinates
        self.roi_coordinates = {'x_temp': 0, 'y_temp': 0, 'x_new': 0, 'y_new': 0, 'clicks': 0}
        
        # Get first frame
        input_clickable, input_annotated, output_annotated, frame_info = self.get_individual_frames(0)
        
        # Set default crop to entire image
        if hasattr(self, 'current_frames') and self.current_frames:
            height, width = self.current_frames['input_rgb'].shape[:2]
            # Set ROI coordinates to cover entire image
            self.roi_coordinates = {
                'x_temp': 0,
                'y_temp': 0,
                'x_new': width,
                'y_new': height,
                'clicks': 2,  # Set to 2 so it's treated as complete selection
            }
            
            # Create default crop previews
            input_img = self.current_frames['input_rgb']
            output_img = self.current_frames['output_rgb']
            input_crop_preview = Image.fromarray(input_img)
            output_crop_preview = Image.fromarray(output_img)
            
            # Create sections for full image annotation
            sections = [((0, 0, width, height), "Crop Region")]
            input_annotated = (input_img, sections)
            output_annotated = (output_img, sections)
            
            return (f"Videos loaded successfully! Total frames: {self.total_frames}", 
                    input_clickable, input_annotated, output_annotated, frame_info, 
                    gr.Slider(minimum=0, maximum=self.total_frames - 1, step=1, value=0, label="🎯 Frame Navigation", interactive=True), 
                    [], f"Default: Full image selected ({width}x{height})",
                    0, 0, width, height, input_crop_preview, output_crop_preview)
        
        return (f"Videos loaded successfully! Total frames: {self.total_frames}", 
                input_clickable, input_annotated, output_annotated, frame_info, 
                gr.Slider(minimum=0, maximum=self.total_frames - 1, step=1, value=0, label="🎯 Frame Navigation", interactive=True), 
                [], "", 0, 0, 0, 0, None, None)
    
    def get_frame(self, cap, idx):
        """Retrieve frame at specific index"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame

    def get_individual_frames(self, frame_idx):
        """Get individual input and output frames"""
        if not self.cap_input or not self.cap_output:
            return None, None, None, "No videos loaded"
            
        frame1 = self.get_frame(self.cap_input, frame_idx)
        frame2 = self.get_frame(self.cap_output, frame_idx)
        
        if frame1 is None or frame2 is None:
            return None, None, None, f"Could not get frame {frame_idx}"
            
        # Convert BGR to RGB for display
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        # Store original frames for cropping
        self.current_frames = {
            'input': frame1,
            'output': frame2,
            'input_rgb': frame1_rgb,
            'output_rgb': frame2_rgb
        }
        
        frame_info = f"Frame {frame_idx}/{self.total_frames-1} | Saved pairs: {len(self.saved_pairs)} | Next: {self.prefix}_{self.crop_counter:03d}"
        
        # Check if we have a current ROI selection to maintain
        if hasattr(self, 'roi_coordinates') and self.roi_coordinates.get('clicks', 0) == 2:
            # Maintain current ROI selection
            x_start = min(self.roi_coordinates['x_temp'], self.roi_coordinates['x_new'])
            y_start = min(self.roi_coordinates['y_temp'], self.roi_coordinates['y_new'])
            x_end = max(self.roi_coordinates['x_temp'], self.roi_coordinates['x_new'])
            y_end = max(self.roi_coordinates['y_temp'], self.roi_coordinates['y_new'])
            
            sections = [((x_start, y_start, x_end, y_end), "Crop Region")]
            return frame1_rgb, (frame1_rgb, sections), (frame2_rgb, sections), frame_info
        
        # Return clickable input frame and annotated versions
        return frame1_rgb, (frame1_rgb, []), (frame2_rgb, []), frame_info
    
    def navigate_frame(self, frame_idx):
        """Navigate to specific frame"""
        self.current_frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        return self.get_individual_frames(self.current_frame_idx)
    
    def get_select_coordinates(self, img, evt: gr.SelectData):
        """Handle click coordinates for ROI selection"""
        if not hasattr(self, 'current_frames'):
            return (self.current_frames['input_rgb'], []), (self.current_frames['output_rgb'], []), 0, 0, 0, 0, "No frame loaded", None, None
        
        sections = []
        # Update new coordinates
        self.roi_coordinates['clicks'] += 1
        self.roi_coordinates['x_temp'] = self.roi_coordinates['x_new']
        self.roi_coordinates['y_temp'] = self.roi_coordinates['y_new']
        self.roi_coordinates['x_new'] = evt.index[0]
        self.roi_coordinates['y_new'] = evt.index[1]
        
        # Compare start end coordinates
        x_start = self.roi_coordinates['x_new'] if (self.roi_coordinates['x_new'] < self.roi_coordinates['x_temp']) else self.roi_coordinates['x_temp']
        y_start = self.roi_coordinates['y_new'] if (self.roi_coordinates['y_new'] < self.roi_coordinates['y_temp']) else self.roi_coordinates['y_temp']
        x_end = self.roi_coordinates['x_new'] if (self.roi_coordinates['x_new'] > self.roi_coordinates['x_temp']) else self.roi_coordinates['x_temp']
        y_end = self.roi_coordinates['y_new'] if (self.roi_coordinates['y_new'] > self.roi_coordinates['y_temp']) else self.roi_coordinates['y_temp']
        
        if self.roi_coordinates['clicks'] % 2 == 0:
            # Both start and end point get
            sections.append(((x_start, y_start, x_end, y_end), "Crop Region"))
            width = x_end - x_start
            height = y_end - y_start
            
            # Get both frame images
            input_img = self.current_frames['input_rgb']
            output_img = self.current_frames['output_rgb']
            
            # Create crop previews
            input_crop_preview = None
            output_crop_preview = None
            if width > 0 and height > 0:
                input_crop = input_img[y_start:y_end, x_start:x_end]
                output_crop = output_img[y_start:y_end, x_start:x_end]
                
                # Convert to PIL Images for display
                input_crop_preview = Image.fromarray(input_crop)
                output_crop_preview = Image.fromarray(output_crop)
            
            # Return annotated versions of both frames with the same ROI
            input_annotated = (input_img, sections)
            output_annotated = (output_img, sections)
            
            return input_annotated, output_annotated, x_start, y_start, width, height, f"ROI Selected: ({x_start}, {y_start}) to ({x_end}, {y_end}) | Size: {width}x{height}", input_crop_preview, output_crop_preview
        else:
            point_width = max(int(img.shape[0]*0.02), 10)  # Smaller point for better visibility
            sections.append(((self.roi_coordinates['x_new'], self.roi_coordinates['y_new'], 
                              self.roi_coordinates['x_new'] + point_width, self.roi_coordinates['y_new'] + point_width),
                            "Click second point for crop region"))
            
            # Get both frame images
            input_img = self.current_frames['input_rgb']
            output_img = self.current_frames['output_rgb']
            
            # Return annotated versions of both frames with the first point indicator
            input_annotated = (input_img, sections)
            output_annotated = (output_img, sections)
            
            return input_annotated, output_annotated, 0, 0, 0, 0, f"First point: ({self.roi_coordinates['x_new']}, {self.roi_coordinates['y_new']}) - Click second point", None, None
    
    def process_image_selection(self, image_editor_data):
        """Process selection from ImageEditor component"""
        if not image_editor_data or not hasattr(self, 'current_frames'):
            return "No selection made or no frame loaded.", "", 0, 0, 0, 0
        
        # Try to extract selection coordinates from the ImageEditor
        # The exact format depends on how the user interacted with the image
        try:
            # ImageEditor returns edited image with selection/annotations
            # We need to detect the selection area
            if 'composite' in image_editor_data:
                edited_image = image_editor_data['composite']
            else:
                edited_image = image_editor_data
                
            # For now, return instruction to use manual coordinates
            return "Use the coordinate boxes below to specify your crop region, or draw on the image above.", "", 0, 0, 100, 100
            
        except Exception as e:
            return f"Error processing selection: {str(e)}", "", 0, 0, 100, 100
    
    def crop_and_save(self, x, y, w, h):
        """Crop and save the selected region using x,y,w,h format"""
        if not hasattr(self, 'current_frames'):
            return "No frame loaded. Please navigate to a frame first.", None
            
        if w <= 0 or h <= 0:
            return "Invalid crop dimensions. Please select a valid region by clicking two points on the image.", None
            
        x, y, w, h = int(x), int(y), int(w), int(h)
        x2, y2 = x + w, y + h
        
        # Get current frames (they are already at original resolution)
        frames = self.current_frames
        input_frame = frames['input']
        output_frame = frames['output']
        
        # Crop both frames using the same coordinates
        input_crop = input_frame[y:y2, x:x2]
        output_crop = output_frame[y:y2, x:x2]
        
        if input_crop.size == 0 or output_crop.size == 0:
            return "Failed to extract crop region. Please check coordinates.", None
            
            # Save crops with 0-indexed naming
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
            'crop_coords': (x, y, x2, y2)
        }
        self.saved_pairs.append(pair_info)
        
        result_msg = f"✓ Saved pair {self.crop_counter:03d}: {input_filename}, {output_filename}\n"
        result_msg += f"Frame: {self.current_frame_idx}, Region: ({x}, {y}) to ({x2}, {y2}) | Size: {w}x{h}"
        
        self.crop_counter += 1
        
        return result_msg, self.get_session_summary()
    
    def auto_crop_every_10th_frame(self):
        """Automatically crop full frames for every 10th frame"""
        if not hasattr(self, 'current_frames') or not self.cap_input or not self.cap_output:
            return "No videos loaded. Please load videos first.", None
            
        if self.total_frames == 0:
            return "No frames available to process.", None
            
        # Get frame dimensions from first frame
        frame1 = self.get_frame(self.cap_input, 0)
        if frame1 is None:
            return "Could not read frame to get dimensions.", None
            
        height, width = frame1.shape[:2]
        
        # Process every 10th frame
        processed_frames = []
        for frame_idx in range(0, self.total_frames, 10):
            # Get frames at this index
            input_frame = self.get_frame(self.cap_input, frame_idx)
            output_frame = self.get_frame(self.cap_output, frame_idx)
            
            if input_frame is None or output_frame is None:
                continue
                
            # Save full frames with 0-indexed naming
            input_filename = f"{self.prefix}_input_{self.crop_counter:03d}.png"
            output_filename = f"{self.prefix}_output_{self.crop_counter:03d}.png"
            
            input_path = os.path.join(self.temp_dir, input_filename)
            output_path = os.path.join(self.temp_dir, output_filename)
            
            cv2.imwrite(input_path, input_frame)
            cv2.imwrite(output_path, output_frame)
            
            # Store pair info
            pair_info = {
                'frame': frame_idx,
                'input_file': input_filename,
                'output_file': output_filename,
                'crop_coords': (0, 0, width, height)
            }
            self.saved_pairs.append(pair_info)
            processed_frames.append(frame_idx)
            self.crop_counter += 1
        
        result_msg = f"✓ Auto-cropped {len(processed_frames)} frames (every 10th frame)\n"
        result_msg += f"Frames processed: {processed_frames}\n"
        result_msg += f"Full frame size: {width}x{height}"
        
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
    
    def crop_wrapper(x, y, w, h):
        return tool.crop_and_save(x, y, w, h)
    
    def auto_crop_wrapper():
        return tool.auto_crop_every_10th_frame()
    
    def download_wrapper():
        return tool.create_download_package()
    
    def roi_selection_wrapper(img, evt: gr.SelectData):
        return tool.get_select_coordinates(img, evt)
    
    # Custom CSS for better styling
    css = """
    .crop-container {
        border: 2px dashed #ccc;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }
    .instruction-box {
        background: #f0f8ff;
        border: 1px solid #87ceeb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(title="Video Crop Tool", theme=gr.themes.Soft(), css=css) as iface:
        gr.Markdown("# 🎬 Video Multi-Crop Tool with Interactive Selection")
        gr.Markdown("Upload two videos, navigate through frames, and select regions using drag-and-drop or coordinates.")
        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        input_video = gr.File(label="📥 Input Video", file_types=[".mp4", ".avi", ".mov", ".mkv"])
                    with gr.Column():
                        output_video = gr.File(label="📤 Output Video", file_types=[".mp4", ".avi", ".mov", ".mkv"])
                
                # Video previews side by side (compact)
                with gr.Row():
                    with gr.Column():
                        input_preview = gr.Video(label="🎬 Input Video Preview", interactive=False, height=400)
                    with gr.Column():
                        output_preview = gr.Video(label="🎬 Output Video Preview", interactive=False, height=400)
                
                prefix_input = gr.Textbox(label="🏷️ Filename Prefix", value="pair", placeholder="Enter prefix for output files")
                load_btn = gr.Button("🚀 Load Videos", variant="primary")
        
        status_text = gr.Textbox(label="📊 Status", interactive=False)
        
        gr.Markdown("---")
        gr.Markdown("## 🎯 Frame-by-Frame Cropping")
        
        # ROI Selection instructions in a single row
        with gr.Row():
            gr.Markdown("### 🎯 ROI Selection: **Click twice** on the input frame to define crop region | **First click:** Start point | **Second click:** End point | **Same region** will be cropped from both frames")
        
        # Three-column layout for frames
        with gr.Row():
            with gr.Column(scale=1):
                input_frame_clickable = gr.Image(label="📥 Input Frame - Click to select crop region", type="numpy")
            with gr.Column(scale=1):
                input_frame_annotated = gr.AnnotatedImage(
                    label="📥 Input Frame with ROI",
                    color_map={"Crop Region": "#9987FF", "Click second point for crop region": "#f44336"}
                )
            with gr.Column(scale=1):
                output_frame_annotated = gr.AnnotatedImage(
                    label="📤 Output Frame with ROI",
                    color_map={"Crop Region": "#9987FF", "Click second point for crop region": "#f44336"}
                )
                
        # Frame navigation
        with gr.Row():
            frame_slider = gr.Slider(
                minimum=0, maximum=1, step=1, value=0,
                label="🎯 Frame Navigation",
                interactive=True
            )
            frame_info = gr.Textbox(label="📝 Frame Info", interactive=False)
        
        # Bottom row for coordinates and preview
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📐 ROI Coordinates (Auto-filled)")
                with gr.Row():
                    x_input = gr.Number(label="X", value=0, precision=0, interactive=False)
                    y_input = gr.Number(label="Y", value=0, precision=0, interactive=False)
                    w_input = gr.Number(label="Width", value=0, precision=0, interactive=False)
                    h_input = gr.Number(label="Height", value=0, precision=0, interactive=False)
                crop_result = gr.Textbox(label="✅ Crop Result", interactive=False)
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 Cropped Preview")
                with gr.Row():
                    input_crop_preview = gr.Image(label="Input Crop", type="pil", height=150)
                    output_crop_preview = gr.Image(label="Output Crop", type="pil", height=150)
                crop_btn = gr.Button("✂️ Crop & Save", variant="secondary", size="lg")
        
        # Selection info at the bottom
        selection_result = gr.Textbox(label="🎯 Selection Info", interactive=False)
        
        # Auto-crop section
        with gr.Row():
            auto_crop_btn = gr.Button("🚀 Auto-Crop Every 10th Frame (Full Size)", variant="primary", size="lg")
            auto_crop_result = gr.Textbox(label="🤖 Auto-Crop Result", interactive=False)
        
        with gr.Row():
            with gr.Column():
                session_summary = gr.Textbox(label="📋 Session Summary", interactive=False, lines=8)
            
            with gr.Column():
                download_btn = gr.Button("📦 Create Download Package", variant="primary")
                download_file = gr.File(label="⬇️ Download Package")
                download_status = gr.Textbox(label="📦 Package Status", interactive=False)
        
        # Event handlers for video preview
        def update_input_preview(video_file):
            return video_file if video_file else None
            
        def update_output_preview(video_file):
            return video_file if video_file else None
        
        input_video.change(
            update_input_preview,
            inputs=[input_video],
            outputs=[input_preview]
        )
        
        output_video.change(
            update_output_preview,
            inputs=[output_video],
            outputs=[output_preview]
        )
        
        # Event handlers
        load_btn.click(
            load_videos_wrapper,
            inputs=[input_video, output_video, prefix_input],
            outputs=[status_text, input_frame_clickable, input_frame_annotated, output_frame_annotated, frame_info, frame_slider, session_summary, selection_result, x_input, y_input, w_input, h_input, input_crop_preview, output_crop_preview]
        )
        
        frame_slider.change(
            navigate_wrapper,
            inputs=[frame_slider],
            outputs=[input_frame_clickable, input_frame_annotated, output_frame_annotated, frame_info]
        )
        
        # Handle ROI selection on input frame
        input_frame_clickable.select(
            roi_selection_wrapper,
            inputs=[input_frame_clickable],
            outputs=[input_frame_annotated, output_frame_annotated, x_input, y_input, w_input, h_input, selection_result, input_crop_preview, output_crop_preview]
        )
        
        crop_btn.click(
            crop_wrapper,
            inputs=[x_input, y_input, w_input, h_input],
            outputs=[crop_result, session_summary]
        )
        
        auto_crop_btn.click(
            auto_crop_wrapper,
            outputs=[auto_crop_result, session_summary]
        )
        
        download_btn.click(
            download_wrapper,
            outputs=[download_file, download_status]
        )
        
        # Instructions
        with gr.Accordion("📖 Detailed Instructions", open=False):
            gr.Markdown("""
            ### How to use:
            1. **Upload Videos**: Select your input and output video files
            2. **Preview Videos**: Watch the uploaded videos side-by-side to verify they're correct
            3. **Set Prefix**: Choose a prefix for your output files (default: 'pair')
            4. **Load Videos**: Click 'Load Videos' to start frame-by-frame analysis
            5. **Navigate**: Use the slider to move through frames
            
            ### ROI Selection Method:
            
            #### Two-Click Selection (Simple & Intuitive)
            1. **Click first point** on the input frame (top-left of desired region)
            2. **Click second point** on the input frame (bottom-right of desired region)
            3. **ROI rectangle** will appear on the output frame preview
            4. **Same coordinates** will be used to crop both input and output frames
            
            #### Visual Feedback:
            - **Input Frame**: Click here to define crop region
            - **Output Frame**: Shows the selected ROI with colored rectangle
            - **Coordinates**: Auto-populate with X, Y, Width, Height
            - **Status**: Real-time feedback on selection progress
            
            ### Coordinate System:
            - The display shows both videos side-by-side
            - X coordinates: 0 to combined width (left video | right video)  
            - Y coordinates: 0 to height (top to bottom)
            - Crops can be from either video or spanning both
            
            ### Workflow:
            1. Select crop region using drawing or coordinates
            2. Click 'Crop & Save' to extract and save the region
            3. Navigate to different frames and repeat as needed
            4. Click 'Create Download Package' to get all crops and batch CSV
            
            ### Output:
            - Individual image pairs: `prefix_input_001.png`, `prefix_output_001.png`, etc.
            - Batch CSV file: `prefix_batch_list.csv` (ready for main.py --batch)
            - All files packaged in a convenient ZIP download
            """)
    
    return iface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=True
    ) 