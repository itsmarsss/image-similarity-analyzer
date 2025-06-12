# Image Similarity Analysis Tool

A comprehensive image comparison tool that analyzes similarity between original and processed images using multiple computer vision and AI techniques. This tool is particularly useful for evaluating self-swap, body-swap, or other image transformation results.

## 🎯 Overview

This tool provides **three different similarity metrics**:

1. **Pixel Difference Score** - Direct pixel-by-pixel comparison
2. **Embedding Difference Score** - Semantic similarity using Cohere's multimodal AI
3. **Pose Difference Score** - Human pose comparison using MediaPipe

The scores are combined into a weighted composite score for comprehensive image analysis.

## 📄 Project Background

This tool was developed as part of a technical report analyzing image similarity metrics for evaluating AI-generated content, particularly body-swap and self-swap transformations. The project demonstrates the effectiveness of combining multiple computer vision approaches (pixel-level, semantic, and pose-based analysis) to provide comprehensive image comparison capabilities.

## 📊 Scoring System

All scores are normalized to a **[0, 1]** range where:

-   **0.0** = Perfect similarity (identical)
-   **1.0** = Maximum difference

### Score Interpretations (Body-Swap Optimized)

| Score Range | Interpretation      | Visual Indicator |
| ----------- | ------------------- | ---------------- |
| 0.0 - 0.1   | Very similar        | 🟢 Green         |
| 0.1 - 0.2   | Moderately similar  | 🟣 Purple        |
| 0.2 - 0.3   | Somewhat different  | 🟠 Orange        |
| 0.3 - 0.4   | Very different      | 🔴 Red           |
| 0.4 - 1.0   | Extremely different | ⚫ Gray          |

_These ranges are optimized for body-swap analysis with default weights [1.0, 2.5, 1.5]_

## 🚀 Quick Start

### Prerequisites

-   Python 3.11+
-   Cohere API key ([Get one here](https://cohere.com/))

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
python main.py -i input.png -o output.png --cohere-key YOUR_API_KEY
```

#### Batch Processing

```bash
# Process multiple pairs from CSV file
python main.py --batch pairs.csv --cohere-key YOUR_API_KEY

# Process all pairs in directory with specific prefix
python main.py --directory ./images --prefix body_swap --cohere-key YOUR_API_KEY
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

| Argument             | Required | Description                                                           |
| -------------------- | -------- | --------------------------------------------------------------------- |
| **Single Pair Mode** |          |                                                                       |
| `-i, --input`        | ✅\*     | Path to the original image                                            |
| `-o, --output`       | ✅\*     | Path to the processed/swapped image                                   |
| **Batch Processing** |          |                                                                       |
| `--batch`            | ✅\*     | Path to batch CSV file with input_path,output_path columns            |
| `--directory`        | ✅\*     | Directory containing image pairs                                      |
| `--prefix`           | ❌       | Filename prefix for directory mode (default: pair)                    |
| **Common Options**   |          |                                                                       |
| `--cohere-key`       | ✅       | Your Cohere API key                                                   |
| `-w, --weights`      | ❌       | Three weights for [pixel, embedding, pose] (default: [1.0, 2.5, 1.5]) |
| `--output-csv`       | ❌       | Save results to specified CSV file                                    |
| `-v, --verbose`      | ❌       | Verbose output for each pair                                          |

\*One mode required: single pair, batch, or directory

### Results Viewer (`results_viewer.py`)

| Argument        | Required | Description                                   |
| --------------- | -------- | --------------------------------------------- |
| `-r, --results` | ✅       | Path to results CSV file or batch file        |
| `-f, --format`  | ❌       | File format: 'csv' or 'batch' (auto-detected) |
| `-p, --port`    | ❌       | Port for web interface (default: 7860)        |
| `--share`       | ❌       | Create public shareable link                  |

### Helper Tool (`helper_select_and_crop.py`)

| Argument       | Required | Description                          |
| -------------- | -------- | ------------------------------------ |
| `-i, --input`  | ✅       | Path to input video                  |
| `-o, --output` | ✅       | Path to output video                 |
| `-w, --width`  | ❌       | Display window width (default: 1200) |
| `-p, --prefix` | ❌       | Filename prefix (default: pair)      |

## 🔄 Complete Workflow

### 1. Extract Image Pairs from Videos

```bash
python helper_select_and_crop.py -i original.mp4 -o processed.mp4 --prefix body_swap
# Creates: body_swap_input_001.png, body_swap_output_001.png, etc.
# Generates: body_swap_batch_list.csv
```

### 2. Analyze Similarity

```bash
python main.py --batch body_swap_batch_list.csv --cohere-key YOUR_KEY
# Creates: similarity_results_TIMESTAMP.csv
```

### 3. View Results Interactively

```bash
python results_viewer.py -r similarity_results_TIMESTAMP.csv
# Opens web interface at http://localhost:7860
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
├── main.py                      # Main analysis tool with batch processing
├── results_viewer.py            # Web-based results viewer (Gradio)
├── helper_select_and_crop.py    # Video frame extraction tool
├── requirements.txt             # Python dependencies
├── methods/                     # Analysis methods
│   ├── method_pixel_diff.py     # Pixel-based comparison
│   ├── method_embedding.py      # AI embedding comparison
│   ├── method_pose.py           # Pose detection comparison
│   └── method_combine_scores.py # Score combination logic
└── README.md                    # This documentation
```

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

### Single Pair Analysis

```bash
$ python main.py -i input.png -o output.png --cohere-key YOUR_KEY

Processing single pair: input.png -> output.png
Using weights: Pixel=1.0, Embedding=2.5, Pose=1.5

[1/1] Processing: input.png -> output.png
  Pixel Score:     0.0771
  Embedding Score: 0.1496
  Pose Score:      1.0000
  Combined Score:  0.4267
```

### Batch Processing

```bash
$ python main.py --batch body_swap_batch_list.csv --cohere-key YOUR_KEY

Loaded 5 pairs from batch CSV: body_swap_batch_list.csv
Using weights: Pixel=1.0, Embedding=2.5, Pose=1.5
Processing 5 pairs...

[1/5] Processing: body_swap_input_001.png -> body_swap_output_001.png
  ✓ Combined Score: 0.2156
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
  Combined Score  - Mean: 0.2145, Min: 0.1834, Max: 0.2567
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

**3. Pose detection not working**

-   Ensure images contain clear human figures
-   Check image quality and lighting
-   MediaPipe works best with full-body or upper-body poses

**4. Memory issues with large images**

-   Images are automatically resized for processing
-   Consider reducing batch sizes for very large datasets

**5. Gradio viewer not opening**

-   Check if port 7860 is available
-   Try a different port: `python results_viewer.py -r results.csv -p 8080`
-   Ensure gradio is installed: `pip install gradio`

## 🎛️ Customization

### 🎯 Recommended Weight Configurations

Based on extensive testing, here are optimal weight configurations for different use cases:

#### **Body-Swap/Self-Swap Analysis** ⭐ _Default Configuration_

```bash
python main.py --batch pairs.csv --cohere-key KEY -w 1.0 2.5 1.5
```

**Rationale:**

-   **Embedding (2.5x)**: Primary focus on semantic realism and identity preservation
-   **Pose (1.5x)**: Important for human body/gesture preservation
-   **Pixel (1.0x)**: Baseline technical quality assessment

#### **General Image Quality Assessment**

```bash
python main.py --batch pairs.csv --cohere-key KEY -w 1.5 2.0 0.5
```

**Rationale:**

-   **Embedding (2.0x)**: Main measure of content preservation
-   **Pixel (1.5x)**: Important for detecting visual artifacts
-   **Pose (0.5x)**: Less critical for non-human subjects

#### **Compression/Technical Quality Testing**

```bash
python main.py --batch pairs.csv --cohere-key KEY -w 2.0 1.0 0.0
```

**Rationale:**

-   **Pixel (2.0x)**: Primary focus on technical degradation detection
-   **Embedding (1.0x)**: Secondary content preservation check
-   **Pose (0.0x)**: Not relevant for compression analysis

#### **Pose-Critical Analysis** (Dance, Sports, Action)

```bash
python main.py --batch pairs.csv --cohere-key KEY -w 0.5 1.5 2.0
```

**Rationale:**

-   **Pose (2.0x)**: Critical for movement/posture accuracy
-   **Embedding (1.5x)**: Maintains content understanding
-   **Pixel (0.5x)**: Minor importance for pose-focused analysis

### 📊 Weight Selection Guide

| Use Case               | Pixel Weight | Embedding Weight | Pose Weight | Best For                       |
| ---------------------- | ------------ | ---------------- | ----------- | ------------------------------ |
| **Body-Swap Analysis** | 1.0          | **2.5**          | 1.5         | Identity preservation, realism |
| **General Quality**    | 1.5          | **2.0**          | 0.5         | Overall image assessment       |
| **Technical Testing**  | **2.0**      | 1.0              | 0.0         | Compression, artifacts         |
| **Pose Analysis**      | 0.5          | 1.5              | **2.0**     | Movement, gesture accuracy     |

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
