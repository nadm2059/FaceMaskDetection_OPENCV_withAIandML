Here’s a complete README.md updated to fully include **both TensorFlow and PyTorch** training and inference workflows, clearly separated and explained:

````markdown
# Face Mask Detection Project

This project implements face mask detection using deep learning with **both TensorFlow and PyTorch** frameworks. It supports training and inference in both, and includes a Streamlit web app for easy interaction.

---

## Project Structure

- `train_pytorch.py` — Train a MobileNetV2 mask detector using PyTorch.
- `train_tensorflow.py` — Train a MobileNetV2 mask detector using TensorFlow.
- `detect_pytorch_image.py` — PyTorch image-based mask detection.
- `detect_pytorch_webcam.py` — PyTorch real-time webcam mask detection.
- `detect_from_image.py` — TensorFlow image-based mask detection.
- `detect_from_webcam.py` — TensorFlow real-time webcam mask detection.
- `app/app.py` — Streamlit app supporting both TensorFlow and PyTorch backends.
- `models/` — Saved models:
  - `mask_detector_pytorch.pth` (PyTorch)
  - `mask_detector.h5` (TensorFlow)
- `data/train/` — Dataset folder with subfolders:
  - `with_mask/`
  - `without_mask/`

---

## Setup & Installation

Install Python dependencies:

```bash
pip install torch torchvision tensorflow opencv-python numpy pillow streamlit
````

---

## Dataset Preparation

Organize your data as follows:

```
data/
└── train/
    ├── with_mask/
    └── without_mask/
```

Images should be labeled by folder name.

---

## Training

### TensorFlow

Run:

```bash
python train_tensorflow.py
```

* Uses Keras MobileNetV2 pretrained on ImageNet
* Applies data augmentation with `ImageDataGenerator`
* Trains for 5 epochs (default)
* Saves model to `models/mask_detector.h5`

### PyTorch

Run:

```bash
python train_pytorch.py
```

* Uses MobileNetV2 pretrained weights from torchvision
* Applies data augmentation (random crop, horizontal flip)
* Trains for 5 epochs (default)
* Saves model weights to `models/mask_detector_pytorch.pth`

---

## Inference

### TensorFlow

**From Image:**

```bash
python detect_from_image.py path_to_image.jpg
```

**From Webcam:**

```bash
python detect_from_webcam.py
```

### PyTorch

**From Image:**

```bash
python detect_pytorch_image.py path_to_image.jpg
```

**From Webcam:**

```bash
python detect_pytorch_webcam.py
```

---

## Streamlit Web App

Launch the web app for easy testing:

```bash
streamlit run app/app.py
```

* Choose backend: TensorFlow or PyTorch
* Upload images and get mask detection results instantly

---

## Model Details

* **Architecture:** MobileNetV2 pretrained on ImageNet
* **Input:** 224x224 RGB images
* **Output:** Single sigmoid neuron for binary mask/no-mask classification
* **Threshold:** Probability < 0.5 → Mask; ≥ 0.5 → No Mask

---

## Notes

* Ensure your `models/` directory contains the trained models before running inference.
* PyTorch training uses a custom RGB conversion to handle transparency issues.
* TensorFlow training uses Keras data augmentation tools.
* Modify training parameters like epochs and batch size in the training scripts.

---

## License

MIT License

---

## Author

Nadhirah 

