"""
analysis.py

Interactive web-based tool for analyzing similarity data correlations and optimizing weights.
Compares computed similarity scores with human annotations to find optimal weight configurations
using Non-Negative Least Squares (NNLS) optimization.

Features:
- Upload similarity results CSV and human annotation CSV files
- Correlation analysis between computed scores and human judgments
- NNLS weight optimization for selected methods
- Method selection for targeted optimization
- Sample data generation for testing
- Detailed statistics and recommendations

Usage:
    python analysis.py [options]
    
Options:
    -p, --port      Port for web interface (default: 7862)
    --share         Create public shareable link
    --help          Show detailed help message with examples

Examples:
    python analysis.py                    # Default port 7862
    python analysis.py -p 8080            # Custom port
    python analysis.py --share            # Public sharing
    python analysis.py -p 8080 --share    # Custom port with sharing

File Requirements:
    Similarity Data CSV (from main.py output):
    - pixel_score, embedding_score, pose_score, combined_score columns
    - Optional: frame column for matching with annotations
    
    Human Annotation CSV:
    - frame column (0, 1, 2, ...)
    - defect/human_annotation/annotation column with ratings

The tool automatically normalizes annotation values to 0-1 scale and provides
optimized weight recommendations for improved similarity analysis.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import nnls
import gradio as gr
import io
import zipfile
import os
import argparse


def analyze_similarity_data(data_csv_file, annotation_csv_file, selected_methods):
    """Analyze similarity data from uploaded CSV files"""
    if data_csv_file is None:
        return "Please upload a similarity data CSV file.", "", ""
    
    if annotation_csv_file is None:
        return "Please upload a human annotation CSV file.", "", ""
    
    if not selected_methods:
        return "Please select at least one method for NNLS optimization.", "", ""
    
    try:
        # Read the similarity data CSV file
        df_data = pd.read_csv(data_csv_file)
        
        # Read the human annotation CSV file
        df_annotations = pd.read_csv(annotation_csv_file)
        
        # Check required columns in data file
        required_data_cols = ['pixel_score', 'embedding_score', 'pose_score', 'combined_score']
        missing_data_cols = [col for col in required_data_cols if col not in df_data.columns]
        
        if missing_data_cols:
            return f"Missing required columns in data file: {missing_data_cols}", "", ""
        
        # Check required columns in annotation file
        if 'frame' not in df_annotations.columns:
            return "Missing 'frame' column in annotation file.", "", ""
        
        # Determine annotation column (could be 'defect', 'human_annotation', 'annotation', etc.)
        annotation_col = None
        possible_cols = ['defect', 'human_annotation', 'annotation', 'score', 'rating']
        for col in possible_cols:
            if col in df_annotations.columns:
                annotation_col = col
                break
        
        if annotation_col is None:
            return f"No annotation column found. Expected one of: {possible_cols}", "", ""
        
        # Add frame index to data if not present
        if 'frame' not in df_data.columns:
            df_data['frame'] = range(len(df_data))
        
        # Merge data with annotations based on frame
        df = df_data.merge(df_annotations[['frame', annotation_col]], on='frame', how='inner')
        
        # Rename annotation column to standard name
        df['human_annotation'] = df[annotation_col]
        
        # Remove rows with NaN values
        required_cols = ['pixel_score', 'embedding_score', 'pose_score', 'combined_score', 'human_annotation']
        df_clean = df.dropna(subset=required_cols)
        if len(df_clean) == 0:
            return "No valid data rows found after merging and removing NaN values.", "", ""
        
        # Normalize human annotations to 0-1 scale if needed
        min_annotation = df_clean['human_annotation'].min()
        max_annotation = df_clean['human_annotation'].max()
        if max_annotation > 1.0 or min_annotation < 0.0:
            # Normalize to 0-1 scale
            df_clean['human_annotation_normalized'] = (df_clean['human_annotation'] - min_annotation) / (max_annotation - min_annotation)
            annotation_note = f" (normalized from [{min_annotation}, {max_annotation}] to [0, 1])"
        else:
            df_clean['human_annotation_normalized'] = df_clean['human_annotation']
            annotation_note = ""
        
        # 1. Basic statistics
        stats_text = f"Dataset Statistics:\n"
        stats_text += f"Data file rows: {len(df_data)}\n"
        stats_text += f"Annotation file rows: {len(df_annotations)}\n"
        stats_text += f"Merged rows: {len(df)}\n"
        stats_text += f"Valid rows (no NaN): {len(df_clean)}\n"
        stats_text += f"Annotation column used: '{annotation_col}'{annotation_note}\n"
        stats_text += f"Selected methods for NNLS: {selected_methods}\n\n"
        
        stats_text += "Score Ranges:\n"
        for col in ['pixel_score', 'embedding_score', 'pose_score', 'combined_score']:
            min_val = df_clean[col].min()
            max_val = df_clean[col].max()
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            stats_text += f"  {col:>17}: [{min_val:.3f}, {max_val:.3f}] (μ={mean_val:.3f}, σ={std_val:.3f})\n"
        
        # Add human annotation stats
        min_val = df_clean['human_annotation'].min()
        max_val = df_clean['human_annotation'].max()
        mean_val = df_clean['human_annotation'].mean()
        std_val = df_clean['human_annotation'].std()
        stats_text += f"  {'human_annotation':>17}: [{min_val:.3f}, {max_val:.3f}] (μ={mean_val:.3f}, σ={std_val:.3f})\n"
        
        # 2. Compute correlations for each method (use normalized annotations)
        methods = ['pixel_score', 'embedding_score', 'pose_score', 'combined_score']
        correlation_text = "\nCorrelation with human annotations:\n"
        correlation_text += f"{'Method':<17} {'Pearson r':<12} {'Spearman ρ':<12} {'Interpretation'}\n"
        correlation_text += "-" * 65 + "\n"
        
        correlations = {}
        for m in methods:
            pearson_r, pearson_p = pearsonr(df_clean[m], df_clean['human_annotation_normalized'])
            spearman_rho, spearman_p = spearmanr(df_clean[m], df_clean['human_annotation_normalized'])
            
            # Interpretation
            abs_r = abs(pearson_r)
            if abs_r >= 0.7:
                interp = "Strong"
            elif abs_r >= 0.5:
                interp = "Moderate"
            elif abs_r >= 0.3:
                interp = "Weak"
            else:
                interp = "Very weak"
            
            # Mark selected methods
            selected_mark = " ✓" if m in selected_methods else ""
            correlation_text += f"{m:<17} {pearson_r:>8.3f}    {spearman_rho:>8.3f}    {interp}{selected_mark}\n"
            correlations[m] = {'pearson': pearson_r, 'spearman': spearman_rho}
        
        # Find best performing method
        best_method = max(correlations.keys(), key=lambda x: abs(correlations[x]['pearson']))
        correlation_text += f"\nBest performing method: {best_method} (r = {correlations[best_method]['pearson']:.3f})\n"
        correlation_text += f"✓ = Selected for NNLS optimization\n"
        
        # 3. NNLS weight optimization using selected methods only
        optimization_text = "\nWeight Optimization Analysis:\n"
        optimization_text += "=" * 50 + "\n"
        optimization_text += f"Using selected methods: {selected_methods}\n\n"
        
        # Use only selected methods for NNLS
        X = df_clean[selected_methods].values
        y = df_clean['human_annotation_normalized'].values
        
        # Solve for non-negative weights
        weights, residual = nnls(X, y)
        
        if weights.sum() > 0:
            # Normalize weights to sum to 1
            weights_norm = weights / weights.sum()
            
            # Calculate correlation with optimized weights
            optimized_score = X @ weights_norm
            opt_r, _ = pearsonr(optimized_score, y)
            
            optimization_text += f"NNLS Optimization Results:\n"
            for i, col in enumerate(selected_methods):
                optimization_text += f"  {col:<17}: {weights[i]:.4f} (normalized: {weights_norm[i]:.4f})\n"
            optimization_text += f"  Correlation with human: {opt_r:.3f}\n"
            optimization_text += f"  Residual: {residual:.4f}\n"
            
            # Generate command line suggestion for all three methods
            w_dict = dict(zip(selected_methods, weights_norm))
            pixel_w = w_dict.get('pixel_score', 0.0)
            embed_w = w_dict.get('embedding_score', 0.0)
            pose_w = w_dict.get('pose_score', 0.0)
            cmd_weights = f"{pixel_w:.2f} {embed_w:.2f} {pose_w:.2f}"
            
            optimization_text += f"\nRecommended Configuration:\n"
            optimization_text += f"Selected methods: {selected_methods}\n"
            optimization_text += f"Optimized weights: {dict(zip(selected_methods, weights_norm))}\n"
            optimization_text += f"Expected correlation: {opt_r:.3f}\n"
            optimization_text += f"\nSuggested main.py command:\n"
            optimization_text += f"python main.py --batch your_file.csv --cohere-key YOUR_KEY -w {cmd_weights}\n"
            
            # Show weight interpretation
            optimization_text += f"\nWeight Interpretation:\n"
            optimization_text += f"  Pixel weight:     {pixel_w:.3f} ({'High' if pixel_w > 0.4 else 'Medium' if pixel_w > 0.2 else 'Low'} importance)\n"
            optimization_text += f"  Embedding weight: {embed_w:.3f} ({'High' if embed_w > 0.4 else 'Medium' if embed_w > 0.2 else 'Low'} importance)\n"
            optimization_text += f"  Pose weight:      {pose_w:.3f} ({'High' if pose_w > 0.4 else 'Medium' if pose_w > 0.2 else 'Low'} importance)\n"
        else:
            optimization_text += "Failed to find valid weights for selected methods.\n"
            optimization_text += "Try selecting different methods or check your data quality.\n"
        
        return stats_text, correlation_text, optimization_text
        
    except Exception as e:
        return f"Error analyzing files: {str(e)}", "", ""


def create_sample_csv():
    """Create sample CSV files for demonstration"""
    # Generate sample data
    np.random.seed(42)
    n_samples = 50
    
    # Simulate some realistic similarity scores
    pixel_scores = np.random.beta(2, 5, n_samples)  # Skewed towards lower values
    embedding_scores = np.random.beta(3, 3, n_samples)  # More balanced
    pose_scores = np.random.beta(2, 3, n_samples)  # Slightly skewed
    
    # Create combined scores (weighted average)
    combined_scores = 0.3 * pixel_scores + 0.5 * embedding_scores + 0.2 * pose_scores
    
    # Create similarity data DataFrame
    similarity_df = pd.DataFrame({
        'input_path': [f'input_{i:03d}.png' for i in range(n_samples)],
        'output_path': [f'output_{i:03d}.png' for i in range(n_samples)],
        'pixel_score': pixel_scores,
        'embedding_score': embedding_scores,
        'pose_score': pose_scores,
        'combined_score': combined_scores,
        'success': [True] * n_samples,
        'error': [None] * n_samples,
        'frame': range(n_samples)  # Add frame column
    })
    
    # Simulate human annotations (correlated with combined but with noise)
    human_annotations = combined_scores + np.random.normal(0, 0.1, n_samples)
    human_annotations = np.clip(human_annotations, 0, 1)  # Keep in [0,1] range
    
    # Convert to defect scale (0-3) like your annotated.csv
    defect_scores = (human_annotations * 3).round().astype(int)
    
    # Create annotation DataFrame
    annotation_df = pd.DataFrame({
        'frame': range(n_samples),
        'defect': defect_scores
    })
    
    # Save to CSV files
    similarity_path = 'sample_similarity_data.csv'
    annotation_path = 'sample_annotations.csv'
    
    similarity_df.to_csv(similarity_path, index=False)
    annotation_df.to_csv(annotation_path, index=False)
    
    # Create a zip file with both CSVs
    zip_path = 'sample_analysis_data.zip'
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(similarity_path, 'similarity_data.csv')
        zipf.write(annotation_path, 'annotations.csv')
    
    # Clean up individual files
    os.remove(similarity_path)
    os.remove(annotation_path)
    
    return zip_path


def create_interface():
    """Create Gradio interface for similarity analysis"""
    
    with gr.Blocks(title="Similarity Analysis Tool", theme=gr.themes.Soft()) as iface:
        gr.Markdown("# 📊 Similarity Analysis Tool")
        gr.Markdown("Upload a CSV file with similarity scores and human annotations to analyze correlations and optimize weights.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📁 Upload Data")
                data_csv_file = gr.File(
                    label="Upload Similarity Data CSV File",
                    file_types=[".csv"],
                    type="filepath"
                )
                
                annotation_csv_file = gr.File(
                    label="Upload Human Annotation CSV File",
                    file_types=[".csv"],
                    type="filepath"
                )
                
                gr.Markdown("### ⚙️ Select Methods for NNLS Optimization")
                method_checkboxes = gr.CheckboxGroup(
                    choices=["pixel_score", "embedding_score", "pose_score"],
                    value=["pixel_score", "embedding_score", "pose_score"],  # Default: all selected
                    label="Methods to Include",
                    info="Choose which similarity methods to optimize weights for"
                )
                
                analyze_btn = gr.Button("🔍 Analyze Data", variant="primary", size="lg")
                
                gr.Markdown("### 📋 Required CSV Columns:")
                gr.Markdown("""
                **Similarity Data CSV:**
                - `pixel_score`: Pixel difference scores
                - `embedding_score`: Semantic embedding scores  
                - `pose_score`: Pose detection scores
                - `combined_score`: Current combined scores
                - `frame` (optional): Frame numbers for matching
                
                **Human Annotation CSV:**
                - `frame`: Frame numbers (0, 1, 2, ...)
                - `defect` or `human_annotation` or `annotation`: Human ratings
                """)
                
                sample_btn = gr.Button("📝 Generate Sample CSV", variant="secondary")
                sample_file = gr.File(label="Sample CSV Download", visible=False)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 Dataset Statistics")
                stats_output = gr.Textbox(
                    label="Statistics",
                    lines=10,
                    interactive=False
                )
            
            with gr.Column():
                gr.Markdown("### 🔗 Correlation Analysis")
                correlation_output = gr.Textbox(
                    label="Correlations",
                    lines=10,
                    interactive=False
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚖️ Weight Optimization")
                optimization_output = gr.Textbox(
                    label="Optimization Results",
                    lines=15,
                    interactive=False
                )
        
        # Event handlers
        analyze_btn.click(
            analyze_similarity_data,
            inputs=[data_csv_file, annotation_csv_file, method_checkboxes],
            outputs=[stats_output, correlation_output, optimization_output]
        )
        
        def generate_sample():
            sample_path = create_sample_csv()
            return gr.File(value=sample_path, visible=True)
        
        sample_btn.click(
            generate_sample,
            outputs=[sample_file]
        )
        
        # Instructions
        with gr.Accordion("📖 How to Use", open=False):
            gr.Markdown("""
            ### Step-by-Step Guide:
            
            1. **Prepare Your Data**: 
                - Export similarity results from main.py (CSV format)
                - Create a separate human annotation CSV with frame numbers and ratings
                - Annotation values will be automatically normalized to 0-1 scale if needed
            
            2. **Select Methods**:
                - Choose which similarity methods to include in NNLS optimization
                - Default: All methods selected (pixel_score, embedding_score, pose_score)
                - Tip: Uncheck methods with poor correlations for better results
            
            3. **Upload & Analyze**:
                - Click "Upload Similarity Data CSV File" and select your similarity data
                - Click "Upload Human Annotation CSV File" and select your human annotations
                - Click "Analyze Data" to see correlations and optimization results
            
            4. **Interpret Results**:
                - **Statistics**: Overview of your dataset and merge results
                - **Correlations**: How well each method matches human judgment (✓ = selected)
                - **Optimization**: Recommended weights for selected methods only
            
            5. **Apply Recommendations**:
                - Use the suggested weights in your main.py command
                - Example: `python main.py --batch data.csv --cohere-key KEY -w 0.2 0.6 0.2`
            
            ### Method Selection Strategy:
            
            **Include methods that:**
            - Show strong correlation with human annotations (|r| > 0.3)
            - Are reliable and consistent in your use case
            - Complement each other (measure different aspects)
            
            **Exclude methods that:**
            - Have very weak correlations (|r| < 0.1)
            - Are noisy or unreliable in your dataset
            - Are redundant with better-performing methods
            
            ### File Format Examples:
            
            **Similarity Data CSV** (from main.py output):
            ```
            input_path,output_path,pixel_score,embedding_score,pose_score,combined_score,success,error
            input_000.png,output_000.png,0.123,0.456,0.789,0.456,True,
            input_001.png,output_001.png,0.234,0.567,0.890,0.567,True,
            ```
            
            **Human Annotation CSV** (your ratings):
            ```
            frame,defect
            0,0
            1,1
            2,2
            3,0
            ```
            
            ### Sample Data:
            - Click "Generate Sample CSV" to download example data (ZIP with both files)
            - Use this to test the tool or as a template for your own annotations
            
            ### Tips:
            - Frame numbers in both files must match for proper merging
            - Annotation column can be named: defect, human_annotation, annotation, score, or rating
            - Values are automatically normalized to 0-1 scale if outside that range
            - Select 2-3 best-performing methods for optimal results
            - Annotate at least 20-30 image pairs for reliable results
            - Higher correlations (closer to 1.0 or -1.0) indicate better methods
            """)
    
    return iface


def main():
    parser = argparse.ArgumentParser(
        description='Interactive Web-based Similarity Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analysis.py                    # Default port 7862
    python analysis.py -p 8080            # Custom port
    python analysis.py --share            # Public sharing
    python analysis.py -p 8080 --share    # Custom port with sharing

File Requirements:
    Similarity Data CSV (from main.py output):
    - pixel_score, embedding_score, pose_score, combined_score columns
    - Optional: frame column for matching with annotations
    
    Human Annotation CSV:
    - frame column (0, 1, 2, ...)
    - defect/human_annotation/annotation column with ratings

Workflow:
    1. Run similarity analysis: python main.py --batch data.csv --cohere-key KEY
    2. Create human annotations CSV with frame numbers and ratings
    3. Upload both files to this tool for correlation analysis
    4. Use recommended weights in future main.py runs

The tool automatically normalizes annotation values and provides optimized
weight recommendations using Non-Negative Least Squares (NNLS) optimization.
        """
    )
    parser.add_argument('-p', '--port', type=int, default=7862, 
                       help='Port for web interface (default: 7862)')
    parser.add_argument('--share', action='store_true', 
                       help='Create public shareable link for remote access')
    args = parser.parse_args()
    
    print(f"\n📊 Starting Similarity Analysis Tool...")
    print(f"📱 Web interface will be available at:")
    print(f"   Local: http://localhost:{args.port}")
    if args.share:
        print(f"🌐 Public link will be generated for sharing")
    print(f"📈 Upload your similarity data and annotations to optimize weights!")
    
    interface = create_interface()
    interface.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
