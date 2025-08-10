
# Face Mask Detection Project

This project provides a complete pipeline for detecting face masks in images and webcam video using deep learning models built with TensorFlow. It includes:

- Training a MobileNetV2-based mask detector model
- Detecting masks from static images
- Real-time mask detection using webcam input
- A simple Streamlit web app for easy interaction

---

## Project Structure

- `src/train.py` — Script to train the mask detector model using transfer learning with MobileNetV2.
- `src/detect_from_image.py` — Script to run mask detection on a single image from the command line.
- `src/detect_from_webcam.py` — Script for real-time mask detection via webcam.
- `app/app.py` — Streamlit web app for interactive mask detection from uploaded images.
- `models/mask_detector.h5` — Saved trained model (created by `train.py`).

---

## Setup & Requirements

- Python 3.7+
- TensorFlow
- OpenCV (`cv2`)
- NumPy
- Streamlit (for the app)
- PIL (Pillow)

Install dependencies using pip:

```bash
pip install tensorflow opencv-python numpy streamlit pillow
````

---

## Usage

### 1. Train the Model

Make sure your training dataset is organized as:

```
data/
└── train/
    ├── with_mask/
    └── without_mask/
```

Run training:

```bash
python src/train.py
```

This will train the model for 5 epochs and save it to `models/mask_detector.h5`.

---

### 2. Detect Mask from Image

Run detection on a single image via command line:

```bash
python src/detect_from_image.py path_to_image.jpg
```

Example output:

```
Prediction: Mask
```

---

### 3. Real-time Detection from Webcam

Run the real-time webcam detection:

```bash
python src/detect_from_webcam.py
```

Press `q` to quit the webcam window.

---

### 4. Run the Streamlit Web App

Start the app:

```bash
streamlit run app/app.py
```

Upload an image via the web interface and get mask detection results instantly.

---

## Model Details

* Base model: MobileNetV2 pretrained on ImageNet (transfer learning).
* Input size: 224x224 RGB images.
* Binary classification: Mask vs No Mask.
* Output: Sigmoid activation for binary prediction.

---

## Notes

* The threshold for mask detection is 0.5. Predictions below 0.5 indicate a mask, above indicate no mask.
* Modify `train.py` to adjust epochs, batch size, or add data augmentations.
* The Streamlit app currently supports only image upload, but can be extended for webcam input.

---

## License

This project is released under the MIT License.

---

## Author

Nadhirah Michael-Ho 

```

---

