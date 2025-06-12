"""
results_viewer.py

Interactive web-based visualization tool for image similarity analysis results.
Displays image pairs side by side with score breakdowns using Gradio interface.

Usage:
    python results_viewer.py --results similarity_results_20231201_143000.csv
    python results_viewer.py --results pair_batch_list.csv --format batch

Options:
    --results, -r   Path to results file (CSV or batch list)
    --format, -f    File format: 'csv' or 'batch' (auto-detected if not specified)
    --port, -p      Port for web interface (default: 7860)
    --share         Create public shareable link
"""

import gradio as gr
import argparse
import pandas as pd
import os
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np
import io
import base64

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Timeline plots will be disabled.")


def load_csv_results(filepath: str) -> pd.DataFrame:
    """Load results from CSV file."""
    try:
        df = pd.read_csv(filepath)
        # Ensure required columns exist
        required_cols = ['input_path', 'output_path', 'success']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return pd.DataFrame()


def load_batch_file(filepath: str) -> pd.DataFrame:
    """Load pairs from batch CSV file (for when results aren't computed yet)."""
    try:
        df = pd.read_csv(filepath)
        # Add empty score columns for batch files
        if 'pixel_score' not in df.columns:
            df['pixel_score'] = None
            df['embedding_score'] = None
            df['pose_score'] = None
            df['combined_score'] = None
            df['success'] = None
            df['error'] = None
        return df
    except Exception as e:
        print(f"Error loading batch file: {e}")
        return pd.DataFrame()


def detect_format(filepath: str) -> str:
    """Auto-detect file format based on content."""
    try:
        df = pd.read_csv(filepath)
        if 'pixel_score' in df.columns or 'success' in df.columns:
            return 'csv'
        else:
            return 'batch'
    except:
        return 'csv'


def load_image_safe(path: str) -> Optional[Image.Image]:
    """Safely load an image file."""
    try:
        if os.path.exists(path):
            return Image.open(path)
        else:
            # Create a placeholder image
            img = Image.new('RGB', (300, 200), color='gray')
            return img
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        # Return a placeholder
        img = Image.new('RGB', (300, 200), color='red')
        return img


def format_score_info(row: pd.Series) -> str:
    """Format score information as HTML."""
    if pd.isna(row.get('success')) or row.get('success') is None:
        return """
        <div style="padding: 15px; background-color: #f0f0f0; border-radius: 8px;">
            <h3 style="color: #666; margin-top: 0;">⏳ Not Analyzed</h3>
            <p>Scores not computed yet.<br>
            Run analysis first:<br>
            <code>python main.py --batch &lt;file&gt; --cohere-key &lt;key&gt;</code></p>
        </div>
        """
    elif row['success']:
        # Successful analysis
        pixel = row['pixel_score']
        embedding = row['embedding_score'] 
        pose = row['pose_score']
        combined = row['combined_score']
        
        # Determine overall similarity level
        if combined <= 0.1:
            level = "Very Similar"
            color = "#28a745"
        elif combined <= 0.2:
            level = "Moderately Similar"
            color = "#6f42c1"
        elif combined <= 0.3:
            level = "Somewhat Different"
            color = "#fd7e14"
        elif combined <= 0.4:
            level = "Very Different"
            color = "#dc3545"
        else:
            level = "Extremely Different"
            color = "#6c757d"
            
        return f"""
        <div style="padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
            <h3 style="color: {color}; margin-top: 0;">📊 Similarity Scores</h3>
            <div style="margin: 10px 0; color: #000;">
                <strong style="color: #000">Pixel Score:</strong> {pixel:.4f}<br>
                <strong style="color: #000">Embedding Score:</strong> {embedding:.4f}<br>
                <strong style="color: #000">Pose Score:</strong> {pose:.4f}<br>
                <hr style="margin: 10px 0;">
                <strong style="font-size: 1.1em; color: #000;">Combined Score: {combined:.4f}</strong><br>
                <span style="color: {color}; font-weight: bold;">{level}</span>
            </div>
            <div style="font-size: 0.9em; color: #666; margin-top: 15px;">
                <strong style="color: #000;">Interpretation:</strong><br>
                0.0-0.1: Very similar<br>
                0.1-0.2: Moderately similar<br>
                0.2-0.3: Somewhat different<br>
                0.3-0.4: Very different<br>
                0.4-1.0: Extremely different
            </div>
        </div>
        """
    else:
        # Error case
        error_msg = row.get('error', 'Unknown error')
        return f"""
        <div style="padding: 15px; background-color: #f8d7da; border-radius: 8px; border: 1px solid #f5c6cb;">
            <h3 style="color: #721c24; margin-top: 0;">❌ Analysis Failed</h3>
            <p style="color: #721c24;"><strong>Error:</strong> {error_msg}</p>
        </div>
        """


def create_individual_score_plots(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Create separate timeline plots for each score type."""
    plots = {}
    
    if not MATPLOTLIB_AVAILABLE or df.empty:
        return plots
    
    # Filter for successful analyses only
    successful_df = df[df['success'] == True].copy()
    
    if successful_df.empty:
        return plots
    
    # If we have frame information, use it for x-axis, otherwise use pair index
    if 'frame' in successful_df.columns:
        x_values = successful_df['frame']
        x_label = 'Frame Number'
        title_suffix = 'by Frame'
    else:
        x_values = range(len(successful_df))
        x_label = 'Pair Index'
        title_suffix = 'by Pair'
    
    # Define score configurations
    score_configs = {
        'pixel': {
            'data': successful_df['pixel_score'],
            'color': '#2E86AB',
            'title': f'Pixel Similarity Score {title_suffix}',
            'description': 'Measures pixel-level differences (MSE-based)'
        },
        'embedding': {
            'data': successful_df['embedding_score'],
            'color': '#A23B72',
            'title': f'Embedding Similarity Score {title_suffix}',
            'description': 'Measures semantic similarity using AI embeddings'
        },
        'pose': {
            'data': successful_df['pose_score'],
            'color': '#F18F01',
            'title': f'Pose Similarity Score {title_suffix}',
            'description': 'Measures human pose and posture differences'
        },
        'combined': {
            'data': successful_df['combined_score'],
            'color': '#C73E1D',
            'title': f'Combined Similarity Score {title_suffix}',
            'description': 'Weighted combination of all similarity metrics'
        }
    }
    
    # Create individual plots
    for score_name, config in score_configs.items():
        plt.figure(figsize=(12, 6))
        
        # Plot the score line
        plt.plot(x_values, config['data'], color=config['color'], linewidth=3, alpha=0.8)
        plt.fill_between(x_values, config['data'], alpha=0.3, color=config['color'])
        
        # Add color-coded background regions for combined score
        if score_name == 'combined':
            plt.axhspan(0.0, 0.1, alpha=0.15, color='green', label='Very Similar')
            plt.axhspan(0.1, 0.2, alpha=0.15, color='blue', label='Moderately Similar')
            plt.axhspan(0.2, 0.3, alpha=0.15, color='orange', label='Somewhat Different')
            plt.axhspan(0.3, 0.4, alpha=0.15, color='red', label='Very Different')
            plt.axhspan(0.4, 1.0, alpha=0.15, color='gray', label='Extremely Different')
            plt.legend(loc='upper right', fontsize=10)
        
        plt.title(config['title'], fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel('Score (0=Similar, 1=Different)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Set y-axis limits based on actual data range for better visualization
        data_min = config['data'].min()
        data_max = config['data'].max()
        data_range = data_max - data_min
        
        # Add 10% padding above and below the data range
        padding = max(0.05, data_range * 0.1)  # Minimum 0.05 padding
        y_min = max(0, data_min - padding)  # Don't go below 0
        y_max = min(1, data_max + padding)  # Don't go above 1
        
        # If data range is very small, ensure minimum visible range
        if data_range < 0.1:
            center = (data_min + data_max) / 2
            y_min = max(0, center - 0.05)
            y_max = min(1, center + 0.05)
        
        plt.ylim(y_min, y_max)
        
        # Add statistics
        mean_score = config['data'].mean()
        min_score = config['data'].min()
        max_score = config['data'].max()
        std_score = config['data'].std()
        
        stats_text = f"""Statistics:
Mean: {mean_score:.3f}  |  Min: {min_score:.3f}  |  Max: {max_score:.3f}  |  Std: {std_score:.3f}
{config['description']}"""
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=11, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        plots[score_name] = f"data:image/png;base64,{plot_data}"
    
    return plots


def create_viewer_interface(df: pd.DataFrame):
    """Create the Gradio interface for viewing results."""
    
    def update_display(pair_index: int) -> Tuple[Image.Image, Image.Image, str, str]:
        """Update the display for a given pair index."""
        if df.empty or pair_index >= len(df):
            placeholder = Image.new('RGB', (300, 200), color='gray')
            return placeholder, placeholder, "No data", "No pairs loaded"
        
        row = df.iloc[pair_index]
        
        # Load images
        input_img = load_image_safe(row['input_path'])
        output_img = load_image_safe(row['output_path'])
        
        # Create header info
        input_name = os.path.basename(row['input_path'])
        output_name = os.path.basename(row['output_path'])
        header = f"**Pair {pair_index + 1} of {len(df)}**\n\n**Input:** {input_name}\n\n**Output:** {output_name}"
        
        # Create score info
        score_info = format_score_info(row)
        
        return input_img, output_img, header, score_info
    
    def next_pair(current_idx: int) -> int:
        return min(current_idx + 1, len(df) - 1) if not df.empty else 0
    
    def prev_pair(current_idx: int) -> int:
        return max(current_idx - 1, 0)
    
    def jump_to_pair(pair_num: int) -> int:
        if df.empty:
            return 0
        return max(0, min(pair_num - 1, len(df) - 1))
    
    # Create Gradio interface
    with gr.Blocks(title="Image Similarity Results Viewer", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔍 Image Similarity Results Viewer")
        gr.Markdown(f"Loaded **{len(df)}** image pairs for analysis")
        
        # Add individual score plots
        score_plots = create_individual_score_plots(df)
        if score_plots:
            with gr.Accordion("📈 Score Timeline Analysis", open=True):
                # Create tabs for each score type
                with gr.Tabs():
                    if 'combined' in score_plots:
                        with gr.Tab("🎯 Combined Score"):
                            gr.HTML(f'<img src="{score_plots["combined"]}" style="width: 100%; max-width: 1200px; height: auto;">')
                    
                    if 'pixel' in score_plots:
                        with gr.Tab("🖼️ Pixel Score"):
                            gr.HTML(f'<img src="{score_plots["pixel"]}" style="width: 100%; max-width: 1200px; height: auto;">')
                    
                    if 'embedding' in score_plots:
                        with gr.Tab("🧠 Embedding Score"):
                            gr.HTML(f'<img src="{score_plots["embedding"]}" style="width: 100%; max-width: 1200px; height: auto;">')
                    
                    if 'pose' in score_plots:
                        with gr.Tab("🤸 Pose Score"):
                            gr.HTML(f'<img src="{score_plots["pose"]}" style="width: 100%; max-width: 1200px; height: auto;">')
        elif not MATPLOTLIB_AVAILABLE:
            with gr.Accordion("📈 Score Timeline Analysis", open=False):
                gr.Markdown("⚠️ **Timeline plots require matplotlib.** Install with: `pip install matplotlib`")
        else:
            # Check if we have any successful analyses
            successful_count = len(df[df['success'] == True]) if 'success' in df.columns else 0
            if successful_count == 0:
                with gr.Accordion("📈 Score Timeline Analysis", open=False):
                    gr.Markdown("📊 **No analyzed results yet.** Run analysis first to see timeline plots.")
        
        # State to track current pair
        current_pair = gr.State(0)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Navigation controls
                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous", variant="secondary")
                    pair_input = gr.Number(label="Go to Pair", value=1, minimum=1, maximum=len(df) if not df.empty else 1, precision=0)
                    next_btn = gr.Button("Next ➡️", variant="secondary")
                
                # Header info
                header_info = gr.Markdown("", elem_id="header-info")
                
                # Images side by side
                with gr.Row():
                    input_image = gr.Image(label="Input Image", type="pil", height=400)
                    output_image = gr.Image(label="Output Image", type="pil", height=400)
            
            with gr.Column(scale=1):
                # Score panel
                score_panel = gr.HTML(label="Similarity Analysis")
        
        # Event handlers
        def on_pair_change(pair_idx):
            img1, img2, header, scores = update_display(pair_idx)
            return img1, img2, header, scores, pair_idx + 1
        
        def on_next(current_idx):
            new_idx = next_pair(current_idx)
            img1, img2, header, scores = update_display(new_idx)
            return img1, img2, header, scores, new_idx, new_idx + 1
        
        def on_prev(current_idx):
            new_idx = prev_pair(current_idx)
            img1, img2, header, scores = update_display(new_idx)
            return img1, img2, header, scores, new_idx, new_idx + 1
        
        def on_jump(pair_num, current_idx):
            new_idx = jump_to_pair(pair_num)
            img1, img2, header, scores = update_display(new_idx)
            return img1, img2, header, scores, new_idx
        
        # Wire up events
        next_btn.click(
            on_next,
            inputs=[current_pair],
            outputs=[input_image, output_image, header_info, score_panel, current_pair, pair_input]
        )
        
        prev_btn.click(
            on_prev,
            inputs=[current_pair],
            outputs=[input_image, output_image, header_info, score_panel, current_pair, pair_input]
        )
        
        pair_input.submit(
            on_jump,
            inputs=[pair_input, current_pair],
            outputs=[input_image, output_image, header_info, score_panel, current_pair]
        )
        
        # Initialize display
        interface.load(
            on_pair_change,
            inputs=[current_pair],
            outputs=[input_image, output_image, header_info, score_panel, pair_input]
        )
        
        # Add keyboard shortcuts info
        gr.Markdown("""
        ### 🎮 Navigation Tips
        - Use **Previous/Next** buttons or the **Go to Pair** number input
        - Images are displayed at their original aspect ratio
        - Scores are color-coded: 🟢 Similar → ⚫ Different
        """)
    
    return interface


def main():
    parser = argparse.ArgumentParser(description='Interactive Web-based Image Similarity Results Viewer')
    parser.add_argument('-r', '--results', required=True, help='Path to results CSV file')
    parser.add_argument('-f', '--format', choices=['csv', 'batch'], 
                       help='File format (auto-detected if not specified)')
    parser.add_argument('-p', '--port', type=int, default=7860, help='Port for web interface')
    parser.add_argument('--share', action='store_true', help='Create public shareable link')
    args = parser.parse_args()
    
    # Detect format if not specified
    file_format = args.format or detect_format(args.results)
    
    # Load results
    print(f"Loading results from {args.results} (format: {file_format})")
    if file_format == 'csv':
        df = load_csv_results(args.results)
    elif file_format == 'batch':
        df = load_batch_file(args.results)
    else:
        print(f"Unknown format: {file_format}")
        return
    
    if df.empty:
        print("No results found or failed to load file!")
        return
    
    print(f"Loaded {len(df)} pairs")
    
    # Create and launch interface
    interface = create_viewer_interface(df)
    
    print(f"\n🚀 Starting web interface...")
    print(f"📱 Open your browser to view the results")
    if args.share:
        print(f"🌐 Public link will be generated for sharing")
    
    interface.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )


if __name__ == '__main__':
    main() 