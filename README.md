# HAID
Histopathological AI Detection

![HAID Logo](images/HAID.png)

---

## 🧠 Overview

**HAID (Histopathological AI Detection)** is an AI-powered deep learning model developed to detect **breast cancer** in histopathological images. It’s designed to assist **medical professionals** with faster, more consistent diagnoses, empowering hospitals and pathology labs with intelligent diagnostic tools.

---

## 📂 Dataset

- **Images:** Over 250,000 high-resolution histopathological samples  
- **Classes:** Binary classification: `Normal` vs. `Cancer`  
- **Image Size:** All images resized to `150x150` pixels  
- **Source:** Private dataset (not publicly shared for privacy & ethical reasons)  
- **Sample Images:**  
  ![Sample Normal](images/sample_class0.jpg) ![Sample Cancer](images/sample_class1.jpg)

---

## 🧱 Model Architecture

- **Base Model:** `EfficientNetB0` (pretrained on ImageNet)
- **Custom Layers:**
  - `GlobalAveragePooling2D`
  - `BatchNormalization`
  - `Dense(256, ReLU)` + `Dropout(0.5)`
  - `BatchNormalization`
  - `Dense(128, ReLU)`
  - `Dense(1, Sigmoid)`
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam (`1e-4` initial, `1e-5` fine-tuning)

---

## 🏋️ Training Details

- **Augmentation Techniques:**
  - Rotation ±25°
  - Shift (Width & Height)
  - Brightness & Zoom range
  - Horizontal & Vertical Flips
  - Shear & Channel shift
- **Epochs:** 30 (initial) + 20 (fine-tuning) = 50 total  
- **Split:** 85% Training / 15% Validation  
- **Early Stopping & LR Reduction:** Enabled  
- **Hardware (AWS SageMaker):**
  - GPU: NVIDIA Tesla T4 (16GB)
  - CPU: Intel Xeon @ 2.50GHz

---

## 📊 Evaluation

- **Validation Accuracy:** ~80% *(to be updated after final evaluation)*  
- **Classification Report:**  
  | Class   | Precision | Recall | F1-Score |
  | ------- | --------- | ------ | -------- |
  | Normal  | 78.3%     | 80.0%  | 79.1%    |
  | Cancer  | 81.2%     | 79.5%  | 80.3%    |

- **Grad-CAM Visualization:**  
  ![Heatmap](images/heatmap_output.jpg)

---

## ⚙️ How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/assemsabry/HAID
   cd HAID

