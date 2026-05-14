# 🍎 Apple Detection & Freshness Classification

This project implements a multi-stage computer vision pipeline. It first detects apples within a complex scene using **YOLOv8**, crops them, and then passes those localized images through a custom **Convolutional Neural Network (CNN)** to classify them as **Fresh** or **Stale**.

## 🚀 Features

* **Object Detection:** Powered by `ultralytics` YOLOv8 to locate apples in any high-resolution image.
* **Automated Cropping:** Automatically extracts detected apples and saves them for quality analysis.
* **Deep Learning Classifier:** A custom 6-layer CNN built with TensorFlow/Keras for binary freshness classification.
* **Dataset Integration:** Seamless integration with Roboflow for YOLO training and Google Drive for CNN dataset management.
* **End-to-End Inference:** A complete script to take a raw image and output the localized apple's freshness and confidence score.

## 📂 Project Structure

```text
Apple-Quality-AI/
├── Apple_Detection_CNN.ipynb   # Full Google Colab notebook
├── README.md                   # Project documentation
├── runs/detect/train2/         # YOLO training results (Confusion matrix, results.png)
└── models/
    ├── YOLO_model2.pt          # Saved YOLO weights
    └── CNN_model.h5            # Saved Keras freshness classifier

```

---

## 🛠️ Installation & Setup

### 1. Install Required Libraries

```bash
pip install ultralytics roboflow tensorflow opencv-python scikit-learn

```

### 2. Dataset Access

* **YOLOv8:** Data is fetched from Roboflow (Apple Detection project).
* **CNN:** Data is expected in Google Drive under `minar project dataset/train` with subfolders `fresh_apple` and `stale_apple`.

---

## 🏗️ Pipeline Architecture

### Stage 1: Apple Detection (YOLOv8)

Locates apples in the input image.

* **Model:** YOLOv8 Nano (`yolov8n.yaml`).
* **Training:** 50 epochs, 640px image size.

### Stage 2: Quality Classification (CNN)

Determines if the detected apple is "Fresh" or "Stale."

* **Input Size:** $255 \times 255 \times 3$.
* **Layers:** 6 Convolutional layers (32 to 128 filters) + Max Pooling.
* **Output:** Sigmoid activation for binary classification.

---

## 🎮 Usage & Inference

The final pipeline performs the following steps:

1. **Inference:** YOLOv8 identifies bounding boxes in `Image1.jpg`.
2. **Preprocessing:** `cv2` crops the bounding box coordinates.
3. **Classification:** The CNN processes the crop:
```python
predictions = model2.predict(input_image)
predicted_class = "Fresh" if predictions[0][0] > 0.5 else "Stale"

```



### Sample Prediction Output:

* **Localized Object:** Apple
* **Condition:** Fresh
* **Confidence:** 0.9842

---

## 📊 Training Performance

The notebook includes automated plotting for:

* **Confusion Matrices:** To evaluate YOLOv8 localization accuracy.
* **Accuracy/Loss Curves:** To monitor CNN freshness training.

---

## 📝 Customization

* **Detection:** Update `data.yaml` to detect other fruits.
* **Sensitivity:** Adjust the `0.5` threshold in the prediction script to change the strictness of "Fresh" vs "Stale" labels.
* **Resolution:** Modify `image_height` and `image_width` to balance speed and accuracy.

> **Note:** This project was developed in **Google Colab** and uses `google.colab.drive` for model persistence. Ensure your directory paths match your Drive structure.
