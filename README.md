# 9517
## Project Structure
```
9517-GroupWork/
9517-GroupWork/
├── TML_Methods/                       # Traditional Machine Learning Methods
│   ├── HSV_KNN/                      # HSV color histogram + KNN classification
│   │   ├── hsv_knn_classification.py # Main script: extract HSV hist features + train/eval KNN
│   │   ├── hsv_features_utils.py     # HSV feature extraction (16 H bins, 8 S, 8 V)
│   │   ├── hsv_knn_reports.txt       # Saved classification_report outputs
│   │   └── hsv_knn_results.ipynb     # Result visualisation (bars, confusion matrix, samples)
│   │
│   ├── HOG_KNN/                      # HOG descriptors + KNN classification
│   │   ├── hog_knn_classification.py # Main script: extract HOG features + train/eval KNN
│   │   ├── hog_features_utils.py     # HOG params (128×64, 9 bins, 8×8, 2×2, L2-Hys)
│   │   ├── hog_knn_reports.txt       # Saved classification_report outputs
│   │   └── hog_knn_results.ipynb     # Confusion matrix, per-class metrics, patch viz
│   │
│   └── HOG_SVM_Detector/             # HOG + Linear SVM sliding-window detector
│       ├── hog_svm_detector.py       # Full pipeline: training, image pyramid, sliding window
│       ├── detector_training.py      # train_detector(): HOG + LinearSVC with class_weight='balanced'
│       ├── detector_evaluation.py    # evaluate_detector(): IoU, TP/FP/FN, precision/recall
│       ├── detection_visualization.ipynb  # Visualization of detection boxes & IoU examples
│       └── requirements.txt          # Python dependencies (scikit-learn, scikit-image, etc.)
│
└── DL_Methods/                   # Deep Learning Methods 需要补充
```

## Desciption
The goal of this project is to develop and compare various computer vision methods for detecting and classifying insects in natural agricultural environments.
1. **TML_Methods**: Contains traditional machine learning approaches  
   - **Color-based Methods (HSV)**:
     - **HSV + KNN**: Uses HSV (Hue–Saturation–Value) color histograms as handcrafted features,
       combined with a K-Nearest Neighbors classifier for 12-class insect classification.
   - **Gradient-based Methods (HOG)**:
     - **HOG + KNN**: Uses Histogram of Oriented Gradients (HOG) descriptors extracted from
       insect patches, followed by a K-Nearest Neighbors classifier for multi-class classification.
     - **HOG + Linear SVM Detector**: Uses HOG features with a Linear Support Vector Machine
       as a binary classifier (insect vs. background), integrated into a sliding-window detector
       with image pyramids and Non-Maximum Suppression (NMS).

2. **DL_Methods**:


## Traditional Machine Learning Technical Details

### HSV+KNN
- **Feature Extraction:** HSV color space histogram
  - H channel: 16 bins
  - S channel: 8 bins
  - V channel: 8 bins
  - Total dimension: 32
- **Classifier:** KNN (k=10, distance weighted)
- **Preprocessing:** StandardScaler normalization

### HOG+KNN
- **Feature Extraction:** HOG (Histogram of Oriented Gradients)
  - Window size: 128×64
  - Orientations: 9
  - Pixels per cell: 8×8
  - Cells per block: 2×2
  - Total dimension: 3780
- **Classifier:** KNN (k=5, distance weighted)
- **Preprocessing:** StandardScaler normalization
The project supports multiple traditional feature extractors and classifiers, which can be flexibly combined to evaluate different configurations for insect classification and detection tasks.
## 可以补充deep learning Technical Details


## Traditional Machine Learning Experimental Results Summary
### 1.Traditional Machine Learning Methods Results
This project evaluates two handcrafted feature extraction methods combined with K-Nearest Neighbors (KNN) for insect image classification.
The following table summarizes the performance of HSV color histograms and HOG descriptors on the test set.
| Feature Type | Feature Dim | Classifier | Accuracy | Precision | Recall | F1 Score |
|---------|------------|------------|----------|-----------|---------|-----------|
| HSV Histogram | 32 | KNN (k=10) | 0.24 | 0.24 | 0.24 | 0.23 |
| HOG Descriptor | ~3780 | KNN (k=5) | 0.16 | 0.31 | 0.15 | 0.14 |
### 2. HSV + KNN Classifier Results
- `hsv_knn_classification_report.txt` - Classification report with metrics
- `hsv_knn_confusion_matrix.png` - **Confusion Matrix - HSV + KNN**
- `hsv_knn_patches_visualization.png` - **Insect Patch Samples (HSV+KNN)**

**Performance Summary:**
- Overall Accuracy: 24%
- Feature Dimension: 32 (HSV color histogram)
- KNN Parameters: k=10, distance weighted

### 3. HOG + KNN Classifier Results
- `hog_knn_classification_report.txt` - Classification report with metrics
- `hog_knn_confusion_matrix.png` - **Confusion Matrix - HOG + KNN**
- `hog_feature_visualization.png` - **HOG Feature Visualization**

**Performance Summary:**
- Overall Accuracy: 16%
- Feature Dimension: 3780 (HOG features)
- KNN Parameters: k=5, distance weighted
- HOG Parameters: orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)

### 4. Performance Comparison
- `classifier_performance_comparison.png` - **HSV+KNN vs HOG+KNN Classification Performance Comparison**
  - Precision Comparison (by class)
  - Recall Comparison (by class)
  - F1-Score Comparison (by class)
  - Overall Accuracy Comparison

**Key Findings:**
- HSV+KNN (24% accuracy) outperforms HOG+KNN (16% accuracy)
- HSV color features are more effective for this insect classification task
- Color is an important identifying feature for insects

### 5. YOLO Annotation Visualization
- `yolo_annotation_good_example_1.png` - **YOLO Good Example 1 (Multiple insects clearly labeled)**
- `yolo_annotation_good_example_2.png` - **YOLO Good Example 2 (Multiple insects clearly labeled)**
- `yolo_annotation_bad_example_1.png` - **YOLO Bad Example 1 (Incomplete or excessive labels)**
- `yolo_annotation_bad_example_2.png` - **YOLO Bad Example 2 (Incomplete or excessive labels)**

### Deep Learning Methods Results 补充


## Installation and Setup

### Requirements
To run this project, you'll need Python 3.8+ and the following dependencies:

```bash
# For ML methods
pip install numpy pandas scikit-learn matplotlib opencv-python seaborn tqdm 

# For DL methods 补充
```
### Dataset

The project uses the AgroPest12 Insect Dataset, which contains 12 insect categories commonly found in agricultural environments.
The dataset is organized into train, valid, and test splits, following the YOLO format:
```
AgroPest12/
├── cleaned/                # Cleaned dataset used for ML and detection experiments
│   ├── train/
│   │   ├── images/         # Training images (cropped insect scenes)
│   │   └── labels/         # YOLO labels: class_id x_center y_center width height
│   ├── valid/
│   │   ├── images/         # Validation images
│   │   └── labels/
│   └── test/
│       ├── images/         # Test images
│       └── labels/
└── data_cleaned.yaml       # Class names + dataset paths for YOLO format
```
### Insect Categories (12 Classes)
```
1. Ants
2. Bees
3. Beetles
4. Caterpillars
5. Earthworms
6. Earwigs
7. Grasshoppers
8. Moths
9. Slugs
10. Snails
11. Wasps
12. Weevils
```
## Running the Code

### Traditional Machine Learning Methods
This project provides three traditional ML pipelines:

HSV Color Histogram + KNN Classification

HOG Descriptor + KNN Classification

HOG + Linear SVM Sliding Window Detector
#### HSV Color Histogram + KNN Classification
This method extracts HSV color histograms (32 dimensions) from YOLO-cropped insect patches and trains a KNN classifier.

```bash
cd TML_Methods/HSV_KNN
python hsv_knn_classification.py
```

To save the classification report:

```bash
hsv_knn_reports.txt'
```

#### HOG + KNN Classification

```bash
cd TML_Methods/HOG_KNN
python hog_knn_classification.py
```

To save the classification report:
```bash
hog_knn_reports.txt'
```
#### HOG + Linear SVM Sliding Window Detector
To Train the detector:
```cd TML_Methods/HOG_SVM_Detector
python hog_svm_detector.py
```
#### Run detection evaluation
```bash
python hog_svm_detector.py --eval
```
### Deep Learning Methods
#### 方法


## Code Documentation

Each module in this project includes clear docstrings and inline comments explaining:
- Feature extraction logic
- Classifier behavior and parameters
- Implementation details and design choices
- References to foundational methods
- Usage examples

All key functions include parameter descriptions and explanatory comments.
