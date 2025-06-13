# Image Similarity Analysis Tool

A comprehensive image comparison tool that analyzes similarity between original and processed images using multiple computer vision and AI techniques. Features flexible method selection, batch processing, interactive results viewing, and video frame extraction tools. Perfect for evaluating AI-generated content, body-swaps, image transformations, and general image quality assessment.

## 🎯 Overview

This tool provides **three different similarity metrics** with **flexible method selection**:

1. **Pixel Difference Score** - Direct pixel-by-pixel comparison using computer vision
2. **Embedding Difference Score** - Semantic similarity using Cohere's multimodal AI
3. **Pose Difference Score** - Human pose comparison using MediaPipe

**Key Features:**
- ✅ **Selective Analysis** - Enable/disable any combination of methods
- ✅ **Flexible Weighting** - Individual or combined weight configuration
- ✅ **Batch Processing** - Process hundreds of image pairs efficiently
- ✅ **Interactive Viewer** - Web-based results exploration with Gradio
- ✅ **Video Tools** - Extract and crop frames from video pairs
- ✅ **Rate Limiting** - Configurable API call delays
- ✅ **Multiple Output Modes** - Verbose, normal, or quiet operation
- ✅ **CSV Export** - Comprehensive results tracking and analysis

The scores are combined into a weighted composite score for comprehensive image analysis.

## 📄 Project Background

This tool was developed as part of a technical report analyzing image similarity metrics for evaluating AI-generated content, particularly body-swap and self-swap transformations. The project demonstrates the effectiveness of combining multiple computer vision approaches (pixel-level, semantic, and pose-based analysis) to provide comprehensive image comparison capabilities.

## 📊 Scoring System

All scores are normalized to a **[0, 1]** range where:

-   **0.0** = Perfect similarity (identical)
-   **1.0** = Maximum difference

### Score Interpretations

| Score Range | Interpretation      | Visual Indicator |
| ----------- | ------------------- | ---------------- |
| 0.0 - 0.1   | Very similar        | 🟢 Green         |
| 0.1 - 0.2   | Moderately similar  | 🟣 Purple        |
| 0.2 - 0.3   | Somewhat different  | 🟠 Orange        |
| 0.3 - 0.4   | Very different      | 🔴 Red           |
| 0.4 - 1.0   | Extremely different | ⚫ Gray          |

_Default weights [1.0, 1.0, 1.0] provide equal weighting. Adjust based on your specific use case._

## 🚀 Quick Start

### Prerequisites

-   Python 3.11+
-   Cohere API key ([Get one here](https://cohere.com/)) - **Only required for embedding analysis**

### Installation

1. **Clone or download the project**

```bash
git clone https://github.com/itsmarsss/image-similarity-analyzer.git
cd image-similarity-analyzer
```

2. **Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Basic Usage

#### Single Image Pair Analysis

```bash
# Full analysis with all methods
python main.py -i input.png -o output.png --cohere-key YOUR_API_KEY

# Skip expensive embedding analysis (no API key needed)
python main.py -i input.png -o output.png --disable-embedding
```

#### Batch Processing

```bash
# Process multiple pairs from CSV file
python main.py --batch pairs.csv --cohere-key YOUR_API_KEY

# Process all pairs in directory with specific prefix
python main.py --directory ./images --prefix body_swap --cohere-key YOUR_API_KEY

# Fast processing without embedding analysis
python main.py --batch pairs.csv --disable-embedding --quiet
```

#### Method Selection Examples

```bash
# Only pixel and pose analysis (no API required)
python main.py --batch pairs.csv --disable-embedding

# Only embedding analysis (semantic similarity)
python main.py --batch pairs.csv --disable-pixel --disable-pose --cohere-key YOUR_KEY

# Custom weights with individual parameters
python main.py --batch pairs.csv --pixel-weight 2.0 --embedding-weight 1.5 --pose-weight 0.5 --cohere-key YOUR_KEY
```

#### Interactive Results Viewer

```bash
# Launch web-based viewer for results
python results_viewer.py -r similarity_results_20231201_143000.csv

# View batch file before analysis
python results_viewer.py -r pairs.csv --format batch
```

## 📋 Command Line Arguments

### Main Analysis Tool (`main.py`)

| Argument                    | Required | Description                                                                    |
| --------------------------- | -------- | ------------------------------------------------------------------------------ |
| **Input/Output Modes**      |          |                                                                                |
| `-i, --input`               | ✅\*     | Path to the original image (single pair mode)                                 |
| `-o, --output`              | ✅\*     | Path to the processed/swapped image (single pair mode)                        |
| `--batch`                   | ✅\*     | Path to batch CSV file with input_path,output_path columns                    |
| `--directory`               | ✅\*     | Directory containing image pairs                                               |
| `--prefix`                  | ❌       | Filename prefix for directory mode (default: pair)                            |
| **Method Control**          |          |                                                                                |
| `--disable-pixel`           | ❌       | Skip pixel difference analysis                                                 |
| `--disable-embedding`       | ❌       | Skip semantic embedding analysis (no API key needed)                          |
| `--disable-pose`            | ❌       | Skip pose detection analysis                                                   |
| **Scoring Options**         |          |                                                                                |
| `-w, --weights`             | ❌       | Three weights for [pixel, embedding, pose] (default: [1.0, 1.0, 1.0])        |
| `--pixel-weight`            | ❌       | Individual weight for pixel score (default: 1.0)                              |
| `--embedding-weight`        | ❌       | Individual weight for embedding score (default: 1.0)                          |
| `--pose-weight`             | ❌       | Individual weight for pose score (default: 1.0)                               |
| **API Configuration**       |          |                                                                                |
| `--cohere-key`              | ❌\*\*   | Cohere API key (required only when embedding analysis enabled)                |
| `--rate-limit-delay`        | ❌       | Delay between API calls in seconds (default: 1.0)                             |
| **Output Options**          |          |                                                                                |
| `--output-csv`              | ❌       | Save results to specified CSV file                                             |
| `--no-auto-save`            | ❌       | Disable automatic timestamped CSV generation for batch processing             |
| `-v, --verbose`             | ❌       | Detailed output showing individual method scores                               |
| `-q, --quiet`               | ❌       | Minimal output (only errors and final results)                                |

\*One mode required: single pair, batch, or directory  
\*\*Required only when embedding analysis is enabled (default)

### Results Viewer (`results_viewer.py`)

| Argument        | Required | Description                                   |
| --------------- | -------- | --------------------------------------------- |
| `-r, --results` | ✅       | Path to results CSV file or batch file        |
| `-f, --format`  | ❌       | File format: 'csv' or 'batch' (auto-detected) |
| `-p, --port`    | ❌       | Port for web interface (default: 7860)        |
| `--share`       | ❌       | Create public shareable link                  |

### Helper Tool (`helper_select_and_crop.py`)

| Argument       | Required | Description                                                    |
| -------------- | -------- | -------------------------------------------------------------- |
| `-i, --input`  | ✅       | Path to input video                                            |
| `-o, --output` | ✅       | Path to output video                                           |
| `-w, --width`  | ❌       | Display window width (default: 1200)                          |
| `-p, --prefix` | ❌       | Filename prefix (default: pair)                               |

### Video Frame Processing (`crop_video_frames.py`)

| Argument         | Required | Description                                                    |
| ---------------- | -------- | -------------------------------------------------------------- |
| `video_file`     | ✅       | Path to input video file                                       |
| `--json-file`    | ❌       | Path to bounding_data.json file (default: bounding_data.json)           |
| `--output-dir`   | ❌       | Output directory for cropped frames (default: cropped_frames) |
| `--bbox-index`   | ❌       | Which bounding box to use if multiple (default: 0)            |
| `--frame-prefix` | ❌       | Prefix for output frame files (default: frame)                |
| `--format`       | ❌       | Output image format: png, jpg (default: png)                  |
| `--skip-missing` | ❌       | Skip frames without bounding box data                          |
| `--verbose`      | ❌       | Show detailed processing information                           |

### File Management (`delete_every_nth.sh`)

| Argument         | Required | Description                                                    |
| ---------------- | -------- | -------------------------------------------------------------- |
| `directory`      | ✅       | Directory containing files to process                          |
| `interval`       | ✅       | Delete every nth file (e.g., 2 = every 2nd file)              |
| `--pattern`      | ❌       | File pattern to match (default: *.png)                        |
| `--inverse`      | ❌       | Keep every nth file instead of deleting every nth file        |
| `--dry-run`      | ❌       | Show what would be deleted without actually deleting           |
| `--force`        | ❌       | Skip confirmation prompt                                       |
| `--verbose`      | ❌       | Show detailed output                                           |

### Score Recalculation (`recalculate_scores.py`)

| Argument              | Required | Description                                                    |
| --------------------- | -------- | -------------------------------------------------------------- |
| `input_csv`           | ✅       | Path to similarity results CSV file                            |
| `--pixel-scale`       | ❌       | Scale factor for pixel scores (default: 1.0)                  |
| `--embedding-scale`   | ❌       | Scale factor for embedding scores (default: 1.0)              |
| `--pose-scale`        | ❌       | Scale factor for pose scores (default: 1.0)                   |
| `--pixel-weight`      | ❌       | Weight for pixel scores (default: 1.0)                        |
| `--embedding-weight`  | ❌       | Weight for embedding scores (default: 1.0)                    |
| `--pose-weight`       | ❌       | Weight for pose scores (default: 1.0)                         |
| `--output`            | ❌       | Output CSV filename                                            |
| `--verbose`           | ❌       | Show detailed statistics and comparisons                       |

### Similarity Analysis & Weight Optimization (`analysis.py`)

| Argument         | Required | Description                                                    |
| ---------------- | -------- | -------------------------------------------------------------- |
| `-p, --port`     | ❌       | Port for web interface (default: 7862)                        |
| `--share`        | ❌       | Create public shareable link for remote access                |
| `--help`         | ❌       | Show detailed help message with examples                       |

## 🔄 Complete Workflow

### 1. Extract Image Pairs from Videos

```bash
# Interactive frame selection and cropping
python helper_select_and_crop.py -i original.mp4 -o processed.mp4 --prefix body_swap
# Creates: body_swap_input_001.png, body_swap_output_001.png, etc.
# Generates: body_swap_batch_list.csv

# OR: Automated frame extraction with bounding boxes
python crop_video_frames.py input_video.mp4 --json-file bounding_data.json --output-dir cropped_frames
```

### 2. Optional: Reduce Frame Count

```bash
# Keep every 3rd frame (reduce by 67%)
./delete_every_nth.sh cropped_frames 3 --inverse --dry-run  # Preview first
./delete_every_nth.sh cropped_frames 3 --inverse           # Execute

# Delete every 2nd frame (reduce by 50%)
./delete_every_nth.sh cropped_frames 2 --dry-run           # Preview first
./delete_every_nth.sh cropped_frames 2                     # Execute
```

### 3. Analyze Similarity

```bash
# Full analysis with all methods
python main.py --batch body_swap_batch_list.csv --cohere-key YOUR_KEY
# Creates: similarity_results_TIMESTAMP.csv

# Fast analysis without embedding (no API key needed)
python main.py --batch body_swap_batch_list.csv --disable-embedding --quiet

# Custom analysis with specific methods and weights
python main.py --batch body_swap_batch_list.csv --disable-pose --pixel-weight 2.0 --embedding-weight 1.5 --cohere-key YOUR_KEY
```

### 4. View Results Interactively

```bash
python results_viewer.py -r similarity_results_TIMESTAMP.csv
# Opens web interface at http://localhost:7860
```

### 5. Optional: Recalculate Scores

```bash
# Experiment with different weights without re-running analysis
python recalculate_scores.py similarity_results_TIMESTAMP.csv \
    --pixel-scale 1.5 --embedding-scale 0.8 \
    --pixel-weight 1.0 --embedding-weight 2.5 --pose-weight 0 \
    --output recalculated_results.csv --verbose
```

### 6. Optional: Optimize Weights with Human Annotations

```bash
# Create human annotation CSV (see format below)
# Then run weight optimization analysis
python analysis.py

# Upload your similarity results CSV and human annotations
# Get optimized weight recommendations based on correlation analysis
```

## 📖 Detailed Method Descriptions

### 1. Pixel Difference Score

-   **Method**: Mean Squared Error (MSE) between pixel values
-   **Normalization**: MSE / (255²) to get [0,1] range
-   **Best for**: Detecting exact visual changes, noise, compression artifacts

### 2. Embedding Difference Score

-   **Method**: Cohere's multimodal embeddings + cosine similarity
-   **Model**: `embed-v4.0` (Cohere v2 API)
-   **Normalization**: `(1 - cosine_similarity) / 2`
-   **Best for**: Semantic similarity, understanding content changes

### 3. Pose Difference Score

-   **Method**: MediaPipe pose detection + landmark comparison
-   **Features**: 33 3D pose landmarks, normalized by shoulder width
-   **Special Cases**:
    -   Both images no pose detected: `0.5` (neutral)
    -   One has pose, other doesn't: `1.0` (maximum difference)
    -   Both have poses: Calculated similarity based on landmark distances
-   **Best for**: Human pose and posture analysis

## 📁 Project Structure

```
.
├── main.py                      # Main analysis tool with flexible method selection
├── results_viewer.py            # Web-based results viewer (Gradio)
├── helper_select_and_crop.py    # Interactive video frame extraction tool
├── crop_video_frames.py         # Automated video frame cropping with JSON bounds
├── delete_every_nth.sh          # File management utility (reduce frame count)
├── recalculate_scores.py        # Score recalculation with different weights
├── analysis.py                  # Weight optimization with human annotations
├── requirements.txt             # Python dependencies
├── methods/                     # Analysis methods
│   ├── method_pixel_diff.py     # Pixel-based comparison
│   ├── method_embedding.py      # AI embedding comparison (Cohere API)
│   ├── method_pose.py           # Pose detection comparison (MediaPipe)
│   └── method_combine_scores.py # Score combination logic with None handling
├── cropped_faces/               # Example output directory
├── bounding_data.json                # Example bounding box data for video cropping
└── README.md                    # This comprehensive documentation
```

## 📊 Weight Optimization with Human Annotations

The `analysis.py` tool provides advanced weight optimization using human annotations to find the best combination of similarity methods for your specific use case.

### Features

- **Correlation Analysis**: Compare computed scores with human judgments
- **NNLS Optimization**: Non-Negative Least Squares weight optimization
- **Method Selection**: Choose which similarity methods to optimize
- **Sample Data Generation**: Create test data for experimentation
- **Interactive Web Interface**: Upload files and view results in browser

### Usage

```bash
# Start the analysis tool
python analysis.py                    # Default port 7862
python analysis.py -p 8080            # Custom port
python analysis.py --share            # Public sharing
```

### Required Data Files

#### 1. Similarity Results CSV (from main.py)
Standard output from the main analysis tool:

```csv
input_path,output_path,pixel_score,embedding_score,pose_score,combined_score,success,error,frame
input_000.png,output_000.png,0.123,0.456,0.789,0.456,True,,0
input_001.png,output_001.png,0.234,0.567,0.890,0.567,True,,1
input_002.png,output_002.png,0.345,0.678,0.901,0.678,True,,2
```

**Required Columns:**
- `pixel_score`: Pixel difference scores (0-1)
- `embedding_score`: Semantic embedding scores (0-1)
- `pose_score`: Pose detection scores (0-1)
- `combined_score`: Current combined scores (0-1)
- `frame` (optional): Frame numbers for matching with annotations

#### 2. Human Annotation CSV
Your manual quality ratings for the same image pairs:

```csv
frame,defect
0,0
1,1
2,2
3,0
4,3
```

**Required Columns:**
- `frame`: Frame numbers matching the similarity data (0, 1, 2, ...)
- `defect` or `human_annotation` or `annotation` or `score` or `rating`: Human quality ratings

**Annotation Scale:**
- Values are automatically normalized to 0-1 scale
- Common scales: 0-1, 0-3, 0-5, 0-10, etc.
- Higher values typically indicate more defects/differences

### Workflow Example

1. **Run Initial Analysis**:
   ```bash
   python main.py --batch pairs.csv --cohere-key YOUR_KEY
   # Generates: similarity_results_TIMESTAMP.csv
   ```

2. **Create Human Annotations**:
   ```bash
   # Manually review image pairs and rate quality
   # Create annotations.csv with frame numbers and ratings
   ```

3. **Optimize Weights**:
   ```bash
   python analysis.py
   # Upload both CSV files in the web interface
   # Select methods to optimize (pixel, embedding, pose)
   # Get correlation analysis and optimized weights
   ```

4. **Apply Optimized Weights**:
   ```bash
   # Use recommended weights from analysis
   python main.py --batch new_data.csv --pixel-weight 0.3 --embedding-weight 0.5 --pose-weight 0.2 --cohere-key YOUR_KEY
   ```

### Analysis Output

The tool provides three types of analysis:

#### Dataset Statistics
- File merge results and data quality
- Score ranges and distributions
- Annotation normalization details

#### Correlation Analysis
- Pearson and Spearman correlations for each method
- Method performance ranking
- Interpretation of correlation strength

#### Weight Optimization
- NNLS-optimized weights for selected methods
- Expected correlation improvement
- Command-line recommendations for main.py

### Method Selection Strategy

**Include methods that:**
- Show strong correlation with human annotations (|r| > 0.3)
- Are reliable and consistent in your use case
- Complement each other (measure different aspects)

**Exclude methods that:**
- Have very weak correlations (|r| < 0.1)
- Are noisy or unreliable in your dataset
- Are redundant with better-performing methods

## 📋 Data Format Specifications

### Bounding Box Data (bounding_data.json)

For automated video frame cropping with `crop_video_frames.py`, the JSON file should contain normalized bounding box coordinates:

```json
{
  "boundingData": [
    {
      "frameIndex": 0,
      "boundingList": [
        {
          "startX": 0.5666666666666667,
          "startY": 0.50625,
          "endX": 0.9236111111111112,
          "endY": 0.9375,
          "uuid": "fb37d099-bb8c-4e69-9caa-c243b0e03d03"
        }
      ]
    },
    {
      "frameIndex": 1,
      "boundingList": [
        {
          "startX": 0.5847222222222223,
          "startY": 0.50625,
          "endX": 0.9444444444444444,
          "endY": 0.93828125,
          "uuid": "fb37d099-bb8c-4e69-9caa-c243b0e03d03"
        }
      ]
    }
  ]
}
```

**Structure:**
- **boundingData**: Array of frame objects
- **frameIndex**: Integer frame number (0, 1, 2, ...)
- **boundingList**: Array of bounding box objects for that frame
- **Coordinates**: Normalized to 0-1 range (relative to frame dimensions)
  - `startX`: Left edge (0 = left side, 1 = right side)
  - `startY`: Top edge (0 = top, 1 = bottom)
  - `endX`: Right edge (0 = left side, 1 = right side)
  - `endY`: Bottom edge (0 = top, 1 = bottom)
- **uuid**: Unique identifier for tracking bounding boxes

**Multiple Bounding Boxes per Frame:**
```json
{
  "boundingData": [
    {
      "frameIndex": 0,
      "boundingList": [
        {
          "startX": 0.1, "startY": 0.1, "endX": 0.3, "endY": 0.4,
          "uuid": "bbox-1"
        },
        {
          "startX": 0.6, "startY": 0.4, "endX": 0.85, "endY": 0.75,
          "uuid": "bbox-2"
        }
      ]
    }
  ]
}
```

Use `--bbox-index` parameter to select which bounding box to use (default: 0).

## 🎬 Video Frame Selection Helper

### Multi-Crop Tool Features

The `helper_select_and_crop.py` tool now supports extracting multiple image pairs from video sequences:

**Enhanced Features:**

-   Load two videos side-by-side (e.g., original vs. processed)
-   Navigate through frames with trackbar and keyboard shortcuts
-   Select multiple regions of interest (ROI) per session
-   Export numbered pairs: `prefix_input_001.png`, `prefix_output_001.png`, etc.
-   Generate CSV batch file automatically for analysis
-   Session summary with all extracted pairs

**Usage:**

```bash
python helper_select_and_crop.py -i input.mp4 -o output.mp4 --prefix body_swap
```

**Controls:**

-   **Trackbar**: Navigate through video frames
-   **'s' key**: Select ROI (opens selection tool)
-   **'n' key**: Next frame
-   **'b' key**: Previous frame
-   **'r' key**: Reset to frame 0
-   **'q' key**: Quit and show summary

**Output:**

-   Individual image pairs: `body_swap_input_001.png`, `body_swap_output_001.png`, etc.
-   Batch CSV file: `body_swap_batch_list.csv`

## 🖥️ Interactive Results Viewer

The new **Gradio-based web interface** provides a modern, user-friendly way to review analysis results:

### Features

-   **Web-based interface** - Opens automatically in your browser
-   **Side-by-side image comparison** with proper scaling
-   **Color-coded score panels** with detailed breakdowns
-   **Navigation controls** - Previous/Next buttons and direct pair jumping
-   **Status-aware display** - Handles analyzed, unanalyzed, and failed cases
-   **Responsive design** - Works on desktop, tablet, and mobile
-   **Shareable links** - Optional public sharing capability

### Usage Examples

```bash
# Basic viewer
python results_viewer.py -r results.csv

# Custom port
python results_viewer.py -r results.csv -p 8080

# Create shareable public link
python results_viewer.py -r results.csv --share

# View batch file before analysis
python results_viewer.py -r batch_list.csv --format batch
```

## 🔧 Dependencies

-   **numpy** - Numerical computations
-   **opencv-python** - Image processing and video handling
-   **cohere** - AI embeddings API (v2)
-   **mediapipe** - Pose detection
-   **pandas** - Data handling for batch processing
-   **gradio** - Web interface for results viewer
-   **Pillow** - Image handling for viewer

## 🔑 API Key Setup

1. Sign up at [Cohere](https://cohere.com/)
2. Generate an API key from your dashboard
3. Use the key with `--cohere-key` parameter

**Security Note**: Never commit API keys to version control. Consider using environment variables:

```bash
export COHERE_API_KEY="your_key_here"
python main.py --batch pairs.csv --cohere-key $COHERE_API_KEY
```

## 📊 Example Output

### Single Pair Analysis (Full Methods)

```bash
$ python main.py -i input.png -o output.png --cohere-key YOUR_KEY

Configuration:
  Weights: Pixel=1.0, Embedding=1.0, Pose=1.0
  Methods: Pixel=✓, Embedding=✓, Pose=✓
  Rate limit delay: 1.0s
Processing 1 pairs...

[1/1] Processing: input.png -> output.png
  Pixel Score:     0.0771
  Embedding Score: 0.1496
  Pose Score:      1.0000
  Combined Score:  0.4089
```

### Single Pair Analysis (Selective Methods)

```bash
$ python main.py -i input.png -o output.png --disable-embedding --disable-pose

Configuration:
  Weights: Pixel=1.0, Embedding=0.0, Pose=0.0
  Methods: Pixel=✓, Embedding=✗, Pose=✗
  Rate limit delay: 1.0s
Processing 1 pairs...

[1/1] Processing: input.png -> output.png
  Pixel Score:     0.0771
  Combined Score:  0.0771
```

### Batch Processing

```bash
$ python main.py --batch body_swap_batch_list.csv --cohere-key YOUR_KEY

Configuration:
  Weights: Pixel=1.0, Embedding=1.0, Pose=1.0
  Methods: Pixel=✓, Embedding=✓, Pose=✓
  Rate limit delay: 1.0s
Processing 5 pairs...

[1/5] Processing: body_swap_input_001.png -> body_swap_output_001.png
  ✓ Combined Score: 0.2156
  Waiting 1.0 seconds to avoid rate limits...
[2/5] Processing: body_swap_input_002.png -> body_swap_output_002.png
  ✓ Combined Score: 0.1834
...

Results saved to: similarity_results_20231201_143000.csv

============================================================
BATCH PROCESSING SUMMARY
============================================================
Total pairs processed: 5
Successful: 5
Failed: 0

STATISTICS (successful pairs):
  Pixel Score     - Mean: 0.0845, Min: 0.0234, Max: 0.1456
  Embedding Score - Mean: 0.1234, Min: 0.0987, Max: 0.1567
  Pose Score      - Mean: 0.8765, Min: 0.5432, Max: 1.0000
  Combined Score  - Mean: 0.3615, Min: 0.2234, Max: 0.5678
```

### Quiet Mode Processing

```bash
$ python main.py --batch large_dataset.csv --disable-embedding --quiet

Results saved to: similarity_results_20231201_143000.csv
```

## 🛠️ Troubleshooting

### Common Issues

**1. `ModuleNotFoundError: No module named '_lzma'`**

```bash
# On macOS with pyenv:
brew install xz
pyenv install --force 3.11.10
# Recreate virtual environment
```

**2. `'Client' object has no attribute 'embed4'`**

-   This usually means an outdated Cohere library
-   Solution: `pip install --upgrade cohere`

**3. "Error: --cohere-key is required when embedding analysis is enabled"**

-   Either provide a Cohere API key: `--cohere-key YOUR_KEY`
-   Or disable embedding analysis: `--disable-embedding`

**4. Pose detection not working**

-   Ensure images contain clear human figures
-   Check image quality and lighting
-   MediaPipe works best with full-body or upper-body poses

**5. Memory issues with large images**

-   Images are automatically resized for processing
-   Consider reducing batch sizes for very large datasets
-   Use `--disable-embedding` to reduce memory usage

**6. Gradio viewer not opening**

-   Check if port 7860 is available
-   Try a different port: `python results_viewer.py -r results.csv -p 8080`
-   Ensure gradio is installed: `pip install gradio`

**7. Rate limiting issues with Cohere API**

-   Increase delay: `--rate-limit-delay 2.0`
-   Process smaller batches
-   Consider disabling embedding analysis for large datasets

**8. "Error: Cannot disable all analysis methods"**

-   At least one method must remain enabled
-   Enable at least one of: `--disable-pixel`, `--disable-embedding`, `--disable-pose`

### Performance Tips

**For Large Datasets:**
```bash
# Fast processing without API calls
python main.py --batch large_dataset.csv --disable-embedding --quiet

# Reduce API calls with longer delays
python main.py --batch dataset.csv --rate-limit-delay 2.0 --cohere-key YOUR_KEY

# Process in smaller chunks
split -l 100 large_dataset.csv chunk_
for chunk in chunk_*; do
    python main.py --batch $chunk --cohere-key YOUR_KEY
done
```

**For Memory-Constrained Systems:**
```bash
# Minimal memory usage
python main.py --batch dataset.csv --disable-embedding --disable-pose --quiet
```

## 🎛️ Customization

### 🎯 Recommended Weight Configurations

Based on extensive testing, here are optimal weight configurations for different use cases:

#### **Body-Swap/Self-Swap Analysis** ⭐ _Recommended for AI-Generated Content_

```bash
# Using individual weights (recommended)
python main.py --batch pairs.csv --pixel-weight 1.0 --embedding-weight 2.5 --pose-weight 1.5 --cohere-key KEY

# Using combined weights
python main.py --batch pairs.csv -w 1.0 2.5 1.5 --cohere-key KEY
```

**Rationale:**
-   **Embedding (2.5x)**: Primary focus on semantic realism and identity preservation
-   **Pose (1.5x)**: Important for human body/gesture preservation
-   **Pixel (1.0x)**: Baseline technical quality assessment

#### **General Image Quality Assessment**

```bash
python main.py --batch pairs.csv --pixel-weight 1.5 --embedding-weight 2.0 --pose-weight 0.5 --cohere-key KEY
```

**Rationale:**
-   **Embedding (2.0x)**: Main measure of content preservation
-   **Pixel (1.5x)**: Important for detecting visual artifacts
-   **Pose (0.5x)**: Less critical for non-human subjects

#### **Compression/Technical Quality Testing**

```bash
# Disable pose analysis entirely for non-human content
python main.py --batch pairs.csv --pixel-weight 2.0 --embedding-weight 1.0 --disable-pose --cohere-key KEY
```

**Rationale:**
-   **Pixel (2.0x)**: Primary focus on technical degradation detection
-   **Embedding (1.0x)**: Secondary content preservation check
-   **Pose (disabled)**: Not relevant for compression analysis

#### **Pose-Critical Analysis** (Dance, Sports, Action)

```bash
python main.py --batch pairs.csv --pixel-weight 0.5 --embedding-weight 1.5 --pose-weight 2.0 --cohere-key KEY
```

**Rationale:**
-   **Pose (2.0x)**: Critical for movement/posture accuracy
-   **Embedding (1.5x)**: Maintains content understanding
-   **Pixel (0.5x)**: Minor importance for pose-focused analysis

#### **Fast Analysis (No API Required)**

```bash
# Pixel and pose only - no API costs
python main.py --batch pairs.csv --disable-embedding --pixel-weight 1.5 --pose-weight 1.0
```

**Rationale:**
-   **No API calls**: Completely free analysis
-   **Pixel (1.5x)**: Primary technical quality measure
-   **Pose (1.0x)**: Human-specific analysis when applicable

### 📊 Weight Selection Guide

| Use Case               | Pixel Weight | Embedding Weight | Pose Weight | API Required | Best For                       |
| ---------------------- | ------------ | ---------------- | ----------- | ------------ | ------------------------------ |
| **Body-Swap Analysis** | 1.0          | **2.5**          | 1.5         | ✅           | Identity preservation, realism |
| **General Quality**    | 1.5          | **2.0**          | 0.5         | ✅           | Overall image assessment       |
| **Technical Testing**  | **2.0**      | 1.0              | disabled    | ✅           | Compression, artifacts         |
| **Pose Analysis**      | 0.5          | 1.5              | **2.0**     | ✅           | Movement, gesture accuracy     |
| **Fast Analysis**      | **1.5**      | disabled         | 1.0         | ❌           | Quick, free analysis           |

### Extending the Tool

To add new comparison methods:

1. Create a new file in `methods/` directory
2. Implement a function that returns a score in [0,1] range
3. Import and integrate in `main.py`
4. Update the scoring combination logic

## 📈 Performance Tips

-   **Batch processing** is more efficient than individual pairs
-   **Directory mode** automatically finds matching pairs
-   **CSV output** provides structured data for further analysis
-   **Gradio viewer** handles large result sets efficiently
-   Use **appropriate weights** for your specific use case

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the tool.

---

**Made with ❤️ for computer vision research and analysis**
