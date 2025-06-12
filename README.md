# Image Similarity Analysis Tool

A comprehensive image comparison tool that analyzes similarity between original and processed images using multiple computer vision and AI techniques. This tool is particularly useful for evaluating self-swap, face-swap, or other image transformation results.

## 🎯 Overview

This tool provides **three different similarity metrics**:

1. **Pixel Difference Score** - Direct pixel-by-pixel comparison
2. **Embedding Difference Score** - Semantic similarity using Cohere's multimodal AI
3. **Pose Difference Score** - Human pose comparison using MediaPipe

The scores are combined into a weighted composite score for comprehensive image analysis.

## 📄 Project Background

This tool was developed as part of a technical report analyzing image similarity metrics for evaluating AI-generated content, particularly face-swap and self-swap transformations. The project demonstrates the effectiveness of combining multiple computer vision approaches (pixel-level, semantic, and pose-based analysis) to provide comprehensive image comparison capabilities.

## 📊 Scoring System

All scores are normalized to a **[0, 1]** range where:

-   **0.0** = Perfect similarity (identical)
-   **1.0** = Maximum difference

### Score Interpretations

| Score Range | Interpretation      |
| ----------- | ------------------- |
| 0.0 - 0.2   | Very similar        |
| 0.2 - 0.4   | Moderately similar  |
| 0.4 - 0.6   | Somewhat different  |
| 0.6 - 0.8   | Very different      |
| 0.8 - 1.0   | Extremely different |

## 🚀 Quick Start

### Prerequisites

-   Python 3.11+
-   Cohere API key ([Get one here](https://cohere.com/))

### Installation

1. **Clone or download the project**

```bash
git clone https://github.com/itsmarsss/image-similarity-analyzer.git
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

```bash
python main.py -i input.png -o output.png --cohere-key YOUR_API_KEY
```

### Advanced Usage

```bash
# Custom weights for different metrics
python main.py -i input.png -o output.png --cohere-key YOUR_API_KEY -w 2.0 1.0 0.5

# This gives more weight to pixel differences, normal weight to embeddings, less to pose
```

## 📋 Command Line Arguments

| Argument        | Required | Description                                                           |
| --------------- | -------- | --------------------------------------------------------------------- |
| `-i, --input`   | ✅       | Path to the original image                                            |
| `-o, --output`  | ✅       | Path to the processed/swapped image                                   |
| `--cohere-key`  | ✅       | Your Cohere API key                                                   |
| `-w, --weights` | ❌       | Three weights for [pixel, embedding, pose] (default: [1.0, 1.0, 1.0]) |

## 📖 Detailed Method Descriptions

### 1. Pixel Difference Score

-   **Method**: Mean Squared Error (MSE) between pixel values
-   **Normalization**: MSE / (255²) to get [0,1] range
-   **Best for**: Detecting exact visual changes, noise, compression artifacts

### 2. Embedding Difference Score

-   **Method**: Cohere's multimodal embeddings + cosine similarity
-   **Model**: `embed-v4.0`
-   **Normalization**: `(1 - cosine_similarity) / 2`
-   **Best for**: Semantic similarity, understanding content changes

### 3. Pose Difference Score

-   **Method**: MediaPipe pose detection + landmark comparison
-   **Features**: 33 3D pose landmarks, normalized by shoulder width
-   **Special Cases**:
    -   Both images no pose detected: `0.5` (neutral)
    -   One has pose, other doesn't: `1.0` (maximum difference)
    -   Both have poses: Calculated similarity
-   **Best for**: Human pose and posture analysis

## 📁 Project Structure

```
.
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── methods/                   # Analysis methods
│   ├── method_pixel_diff.py   # Pixel-based comparison
│   ├── method_embedding.py    # AI embedding comparison
│   ├── method_pose.py         # Pose detection comparison
│   └── method_combine_scores.py # Score combination logic
├── input.png                  # Sample input image
├── output.png                 # Sample output image
└── README.md                  # This file
```

## 🎬 Video Frame Selection Helper

For video analysis workflows, this project includes a helpful utility script:

### `helper_select_and_crop.py`

This interactive tool helps you extract specific frames and regions from video pairs:

**Features:**

-   Load two videos side-by-side (e.g., original vs. processed)
-   Navigate through frames with a trackbar
-   Select regions of interest (ROI) by drawing rectangles
-   Export cropped frames as `input.png` and `output.png`

**Usage:**

```bash
python helper_select_and_crop.py -i input.mp4 -o output.mp4 --width 1280
```

**Controls:**

-   **Trackbar**: Navigate through video frames
-   **'s' key**: Select ROI (opens selection tool)
-   **'q' key**: Quit without saving

**Workflow Example:**

```bash
# Step 1: Extract frames from videos
python helper_select_and_crop.py -i original_video.mp4 -o processed_video.mp4

# Step 2: Analyze the extracted images
python main.py -i input.png -o output.png --cohere-key YOUR_KEY
```

This tool is particularly useful for:

-   Video face-swap analysis
-   Comparing specific moments in video sequences
-   Creating matched image pairs from video content

## 🔧 Dependencies

-   **numpy** - Numerical computations
-   **opencv-python** - Image processing
-   **cohere** - AI embeddings API
-   **mediapipe** - Pose detection

## 🔑 API Key Setup

1. Sign up at [Cohere](https://cohere.com/)
2. Generate an API key from your dashboard
3. Use the key with `--cohere-key` parameter

**Security Note**: Never commit API keys to version control. Consider using environment variables:

```bash
export COHERE_API_KEY="your_key_here"
python main.py -i input.png -o output.png --cohere-key $COHERE_API_KEY
```

## 📊 Example Output

```bash
$ python main.py -i input.png -o output.png --cohere-key YOUR_KEY

Pose detection: Swapped has pose, original doesn't
Pixel Score:     0.0771
Embedding Score: 0.1496
Pose Score:      1.0000
Combined Score:  1.2267
```

**Interpretation**: The images show moderate pixel similarity, good semantic similarity, but maximum pose difference (one has detectable human pose, the other doesn't).

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

-   Resize images before processing
-   Consider processing in batches for multiple images

## 🎛️ Customization

### Adjusting Weights

The weighting system lets you emphasize different aspects:

```bash
# Emphasize pose differences (useful for dance/sports analysis)
python main.py -i img1.png -o img2.png --cohere-key KEY -w 0.5 0.5 2.0

# Focus on semantic content (useful for style transfer evaluation)
python main.py -i img1.png -o img2.png --cohere-key KEY -w 0.5 2.0 0.5

# Pure pixel comparison (useful for compression quality)
python main.py -i img1.png -o img2.png --cohere-key KEY -w 2.0 0.0 0.0
```

### 🎯 Recommended Weight Configurations

Based on extensive testing, here are optimal weight configurations for different use cases:

#### **Face-Swap/Self-Swap Analysis** ⭐ _Recommended for this project_

```bash
python main.py -i input.png -o output.png --cohere-key KEY -w 1.0 2.5 1.5
```

**Rationale:**

-   **Embedding (2.5x)**: Primary focus on semantic realism and identity preservation
-   **Pose (1.5x)**: Important for human body/gesture preservation
-   **Pixel (1.0x)**: Baseline technical quality assessment

#### **General Image Quality Assessment**

```bash
python main.py -i input.png -o output.png --cohere-key KEY -w 1.5 2.0 0.5
```

**Rationale:**

-   **Embedding (2.0x)**: Main measure of content preservation
-   **Pixel (1.5x)**: Important for detecting visual artifacts
-   **Pose (0.5x)**: Less critical for non-human subjects

#### **Compression/Technical Quality Testing**

```bash
python main.py -i input.png -o output.png --cohere-key KEY -w 2.0 1.0 0.0
```

**Rationale:**

-   **Pixel (2.0x)**: Primary focus on technical degradation detection
-   **Embedding (1.0x)**: Secondary content preservation check
-   **Pose (0.0x)**: Not relevant for compression analysis

#### **Pose-Critical Analysis** (Dance, Sports, Action)

```bash
python main.py -i input.png -o output.png --cohere-key KEY -w 0.5 1.5 2.0
```

**Rationale:**

-   **Pose (2.0x)**: Critical for movement/posture accuracy
-   **Embedding (1.5x)**: Maintains content understanding
-   **Pixel (0.5x)**: Minor importance for pose-focused analysis

### 📊 Weight Selection Guide

| Use Case               | Pixel Weight | Embedding Weight | Pose Weight | Best For                       |
| ---------------------- | ------------ | ---------------- | ----------- | ------------------------------ |
| **Face-Swap Analysis** | 1.0          | **2.5**          | 1.5         | Identity preservation, realism |
| **General Quality**    | 1.5          | **2.0**          | 0.5         | Overall image assessment       |
| **Technical Testing**  | **2.0**      | 1.0              | 0.0         | Compression, artifacts         |
| **Pose Analysis**      | 0.5          | 1.5              | **2.0**     | Movement, gesture accuracy     |

### Extending the Tool

To add new comparison methods:

1. Create a new file in `methods/` directory
2. Implement a function that returns a score in [0,1] range
3. Import and integrate in `main.py`

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the tool.

---

**Made with ❤️ for computer vision research and analysis**
