# SAM-MedSeg-Kit

## 📖 Project Overview

This is a comprehensive experimental framework for the **cross-modal evolutionary evaluation of SAM1/2/3 in medical image segmentation under a zero-shot setting**. The project implements a medical image segmentation pipeline based on the Segment Anything Model series, supporting multiple publicly available medical image datasets.

**Key Features:**
- Supports three versions: SAM1, SAM2, and SAM3
- Integrates multiple medical datasets (ACDC, BUSI, IDRiD, Synapse)
- Automated data preprocessing workflow
- Built-in automatic visual cue extraction (no manual annotation required)
- Comprehensive evaluation metric calculation
- High-quality chart generation

## 📁 Project Structure

```
.
├── README.md # This file
├── SAM_Easy.py # ⭐ Beginner-friendly unified SAM interface
├── 
├── Data Preprocessing Modules
├── ├── ACDC_data_preprocess.py # ACDC dataset preprocessing
├── ├── Synapse_data_preprocess.py # Synapse dataset preprocessing
├── ├── BUSI_data_preprocess.py # BUSI dataset preprocessing
├── ├── IDRiD_data_preprocess.py # IDRiD dataset preprocessing
├── 
├── SAM Segmentation Module
├── ├── ACDC_data_sam1_seg.py # ACDC using SAM1 segmentation
├── ├── ACDC_data_sam2_seg.py # ACDC using SAM2 segmentation
├── ├── ACDC_data_sam3_seg.py # ACDC using SAM3 segmentation
├── ├── BUSI_data_sam1_seg.py # BUSI using SAM1 segmentation
├── ├── BUSI_data_sam2_seg.py # BUSI using SAM2 segmentation
├── ├── BUSI_data_sam3_seg.py # BUSI segmentation using SAM3
├── ├── IDRiD_data_sam1_seg.py # IDRiD segmentation using SAM1
├── ├── IDRiD_data_sam2_seg.py # IDRiD segmentation using SAM2
├── ├── IDRiD_data_sam3_seg.py # IDRiD segmentation using SAM3
├── ├── Synapse_data_sam1_seg.py # Synapse segmentation using SAM1
├── ├── Synapse_data_sam2_seg.py # Synapse segmentation using SAM2
├── └── Synapse_data_sam3_seg.py # Synapse segmentation using SAM3
```


## 🚀 Quick Start

### 1️⃣ Environment Setup

```bash
# Create a virtual environment
conda create -n sam-med python=3.10
conda activate sam-med

# Install Core Dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
- `ultralytics>=8.0.0` - SAM model wrapper
- `opencv-python` - Image processing
- `nibabel` - Medical image format support (.nii.gz)
- `scikit-image` - Morphological algorithms
- `pandas` - Data analysis
- `matplotlib`, `seaborn` - Visualization

### 2️⃣ Download model weights

Create a `Models/` folder in the project root directory and download the following weights:

| Model | Filename | Size | Source |
|------|------- -|------|------|
| SAM1 Large | `sam_l.pt` | 1.2 GB | [Meta AI](https://github.com/facebookresearch/segment-anything) |
| SAM2 Large | `sam2_l.pt` | 429 MB | [Meta AI](https://github.com/facebookresearch/sam2) |
| SAM3 | `sam3.pt` | 3.3 GB | [Meta AI](https://github.com/facebookresearch/segment-anything-3) |

Directory structure:
```
Models/
├── sam2_l.pt
├── sam3.pt
└── sam_l.pt
```

### 3️⃣ Prepare the data

Assuming the raw data has been downloaded to the `DataSets/` directory:

```bash
# Using ACDC as an example
python ACDC_data_preprocess.py

# The preprocessed output is located in DataSets/ACDC_pro/
```

### 4️⃣ Run Segmentation

**Method 1: Use the beginner-friendly `SAM_Easy.py` (Recommended)**

```python
from SAM_Easy import SAMSegmenter

# Create a segmenter instance
segmenter = SAMSegmenter(
    model_type="SAM1", # Optional: SAM1, SAM2, SAM3
    model_path="Models/sam_l.pt" # Model weight path
)

# Segment a single image
result = segmenter.segment_image(
    image_path="path/to/image.png",
    use_auto_prompt=True, # Automatically extract visual prompts
    output_path="output.png"
)
print(f“IoU: {result[‘iou’]:.4f}”)

# Batch segmentation of images in a directory
segmenter.segment_directory(
    input_dir="DataSets/ACDC_pro",
    output_dir="DataRes/ACDC_pro/predictions",
    gt_dir="DataSets/ACDC_pro" # Used to calculate evaluation metrics
)
```

**Method 2: Run the original script directly**

```bash
# Use SAM1 to segment the ACDC dataset
python ACDC_data_sam1_seg.py

# Use SAM2 to segment the BUSI dataset
python BUSI_data_sam2_seg.py

# Use SAM3 to segment the Synapse dataset
python Synapse_data_sam3_seg.py
```


## 📊 Supported Datasets

| Dataset | Modality | Organ/Lesion | Script File |
|--------|------|---------|---------|
| **ACDC** | Cardiac CT | Ventricular Segmentation (RV/MYO/LV) | `ACDC_data_*_seg.py` |
| **BUSI** | Breast Ultrasound | Tumor Detection | `BUSI_data_*_seg.py` |
| **IDRiD** | Retinal RGB | Diabetic Lesion Detection | `IDRiD_data_*_seg.py` |
| **Synapse** | Abdominal CT | Multi-organ Segmentation | `Synapse_data_*_seg.py` |


## 🧠 Core Feature Description

### Automatic Prompt Extraction (Vision-Only)

No additional annotation required; the system automatically generates bounding box prompts using computer vision algorithms:

```python
def extract_prompt_from_vision(img_path):
    “”“
    Automatically extract target regions using morphological operations + Otsu thresholding
    ”“”
    # Gaussian filter → Morphological gradient → Otsu binarization → Contour detection
    # Return: bbox (x, y, w, h) or None
```

### Evaluation Metrics

Supported evaluation metrics:
- **IoU (Intersection over Union)** - Primary metric
- **Dice Coefficient** - Medical imaging standard
- **Hausdorff Distance** - Contour similarity
- **Sensitivity / Specificity** - Clinical metrics

```python
metrics = segmenter.compute_metrics(
    prediction_mask=pred,
    ground_truth_mask=gt
)
# {‘iou’: 0.85, ‘dice’: 0.92, ‘hausdorff’: 15.3, ...}
```

### Batch Processing

Automate processing of large numbers of images and generate reports:

```python
results_df = segmenter.segment_directory(
    input_dir="DataSets/ACDC_pro",
    output_dir="DataRes/ACDC_results",
    gt_dir="DataSets/ACDC_pro",
    save_report=True # Generate CSV report
)
# Output: results_df (contains IoU, Dice, and other metrics for all images)
```

## 📝 Common Workflows

### Workflow 1: Quickly evaluate a new dataset

```bash
# 1. Preprocessing
python {DATASET}_data_preprocess.py

# 2. Segmentation (using the best model, SAM3)
python {DATASET}_data_sam3_seg.py

# 3. View results
# → Prediction results are in DataRes/{DATASET}_pro/sam3_preds/
# → Detailed metrics are in the CSV report
```

### Workflow 2: Three-Model Comparison

```bash
# Run the three versions
python ACDC_data_sam1_seg.py
python ACDC_data_sam2_seg.py
python ACDC_data_sam3_seg.py
```

### Workflow 3: From Scratch (Full Process)

```bash
# 1. Data preprocessing
python ACDC_data_preprocess.py
python BUSI_data_preprocess.py
python IDRiD_data_preprocess.py
python Synapse_data_sam1_seg.py

# 2. Segment each dataset using SAM1
for dataset in ACDC BUSI IDRiD Synapse; do
    python ${dataset}_data_sam1_seg.py
done

# 3. Perform upscaling using SAM2/3
python ACDC_data_sam2_seg.py
python ACDC_data_sam3_seg.py
# ... Other datasets
```


## 🔧 Configuration Adjustments

All script configurations are stored in the `CONFIG` dictionary and can be modified as needed:

```python
CONFIG = {
    "model": "Models/sam_l.pt",        # Model weight path
    "image_dir": "DataSets/ACDC_pro",  # Input directory
    "output_dir": "DataRes/ACDC_pro/sam1_preds",  # Output directory
    "batch_size": 4,                   # GPU batch size
    "device": "cuda:0",                # GPU ID
}
```

**Common Adjustments:**

| Configuration Item | Purpose | Recommended Value |
|--------|------|--------|
| `model` | Model weights | `sam_b.pt` (fast) or `sam_l.pt` (strong) |
| `batch_size` | GPU parallelism | 4-8 (depending on GPU VRAM) |
| `device` | Computing device | `cuda:0` (GPU) or `cpu` (CPU) |
| `iou_threshold` | Prediction confidence | 0.5–0.9 |


## ⚙️ Dependency Version Reference

```
ultralytics>=8.0.0
torch>=2.0.0,<3
torchvision>=0.15.0
opencv-python>=4.8.0
nibabel>=5.1.0
scikit-image>=0.21.0
scipy>=1.11.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
```


## 📊 Output Structure

After execution, the output directory structure is as follows:

```
DataRes/ACDC_pro/sam1_preds/
├── case_001_mask.png # Predicted segmentation mask
├── case_002_mask.png
├── ...
└── metrics_report.csv # Evaluation report
    ├── image_id | IoU | Dice | Hausdorff | Sensitivity | Specificity
    └── ...
```


## 🎓 Tips for Beginners

1. **Start with `SAM_Easy.py`**
   - This is the simplest interface, ideal for rapid prototyping
   - Supports automatic feature extraction; no manual annotation required

2. **Start with SAM1 (fastest speed)**
   - Quickly verify if the workflow is correct
   - Then upgrade to SAM2/3 for better results

3. **Check the log output**
   - All scripts use `pwn.log` to print progress
   - If you encounter issues, check the error log to identify the cause


## 📬 Frequently Asked Questions

**Q1: Model download too slow?**
A: We recommend using a domestic mirror or proxy, or downloading the model in advance and placing it in the `Models/` directory.

**Q2: Not enough GPU VRAM?**
A: Set `batch_size` to 1–2, or use `sam_b.pt` (a smaller model).

**Q3: Poor prediction results?**
A:
- Check if the input image preprocessing is correct (normalization range)
- Try upgrading to SAM2/3
- Adjust the `iou_threshold` parameter

**Q4: How do I use the CPU?**
A: Set `CONFIG[“device”] = “cpu”`, but this will significantly slow down the process.

## 🤝 Feedback and Support

If you have questions, suggestions, or need technical support, please:
- Review the FAQ section of this README
- Check the log information output by the script
- Refer to the original research paper and the official SAM documentation

**Release Notes:**
- v1.0 (2026-05-25): Initial release, supports SAM1/2/3

---

**Enjoy!** 🎉
