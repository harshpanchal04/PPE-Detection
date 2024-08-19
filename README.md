# PPE Detection Project
## Documentation - https://docs.google.com/document/d/1Hw8jjdOXDb3vB7RjMGEXrgPi9q_nZ3mPJyfggcHDfSc/pub
## Overview
This project implements an automated system for detecting Personal Protective Equipment (PPE) in images using YOLOv8. It includes scripts for data preparation, model training, and inference.

## Features
- Conversion of PascalVOC annotations to YOLO format
- Separate models for person detection and PPE detection
- Image cropping based on person detection
- PPE detection on cropped person images
- Comprehensive pipeline for end-to-end processing

## Requirements
- Python 3.8+
- PyTorch 1.7+
- OpenCV
- Ultralytics YOLOv8

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/harshpanchal04/PPE-Detection.git
   cd ppe-detection
   ```

## Usage
1. Prepare your dataset in PascalVOC format.

2. Convert annotations to YOLO format:
   ```
   python scripts/VOC_toyolo.py --input_dir /path/to/dataset --output_dir /path/to/output
   ```

3. Train the models (refer to the documentation for Google Colab training instructions).

4. Run the inference pipeline:
   ```
   python run_pipeline.py /path/to/input_dir /path/to/output_dir /path/to/person_model.pt /path/to/ppe_model.pt
   ```

## Project Structure

Note: This repository only contains the scripts and weights to be used for ppe and person detection.

```
ppe-detection-project/
├── scripts/
│   ├── VOC_toyolo.py
│   ├── separate_labels.py
│   ├── crop_person_images.py
│   └── inference.py
├── weights/
│   ├── bestest_person.pt (inside final directory)
│   └── bestest_ppe.pt (inside final directory)
├── run_pipeline.py
├── requirements.txt
├── README.md
└── documentation.pdf
```

## Documentation
For detailed information about the project, methodology, and results, please refer to the `documentation.pdf` file.

## Contact
For any queries, please open an issue in the GitHub repository.
