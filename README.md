Here's a comprehensive **README.md** for your face mask detection project that covers all your scripts, usage, and setup:

```markdown
# Face Mask Detection Project

This project implements a face mask detection system using deep learning with TensorFlow and MobileNetV2. It includes scripts for training a model, detecting masks from images and webcam video, and a simple web app built with Streamlit.

---

## Features

- Train a mask detection model on labeled images (`with_mask` and `without_mask`).
- Detect face masks in static images (`detect_from_image.py`).
- Real-time mask detection from webcam feed (`detect_from_webcam.py`).
- Streamlit web app for uploading images and predicting mask presence (`app.py`).

---

## Project Structure

```

src/
├── detect\_from\_image.py       # Detect mask from a single image file via command line
├── detect\_from\_webcam.py      # Detect mask from webcam feed in real-time
├── train.py                   # Train the mask detection model using TensorFlow
app/
└── app.py                    # Streamlit web application for mask detection
models/
└── mask\_detector.h5          # Saved trained model (TensorFlow)
data/
└── train/                    # Training data with subfolders 'with\_mask' and 'without\_mask'

````

---

## Setup Instructions

### Requirements

- Python 3.7+
- TensorFlow
- OpenCV
- NumPy
- Streamlit (for web app)
- PIL (Pillow)

Install required packages using pip:

```bash
pip install tensorflow opencv-python numpy streamlit pillow
````

### Prepare Dataset

* Organize your training images inside `data/train` with two subfolders:

  * `with_mask/`
  * `without_mask/`

Each folder should contain relevant images.

---

## Usage

### 1. Train the Model

Train the model with data augmentation and validation split:

```bash
python src/train.py
```

This saves the trained model as `models/mask_detector.h5`.

---

### 2. Detect Mask from Image

Run detection on an image file via command line:

```bash
python src/detect_from_image.py path/to/image.jpg
```

Output:

```
Prediction: Mask
```

or

```
Prediction: No Mask
```

---

### 3. Detect Mask from Webcam

Start webcam mask detection (press 'q' to quit):

```bash
python src/detect_from_webcam.py
```

A window will open showing the webcam feed with the predicted label ("Mask" or "No Mask") overlayed.

---

### 4. Run Streamlit Web App

Launch the web app for easy image upload and mask detection:

```bash
streamlit run app/app.py
```

Upload an image and get instant prediction with a user-friendly interface.

---

## Notes

* The model is based on MobileNetV2 pretrained on ImageNet with the top layer replaced for binary classification.
* Image input size is 224x224 pixels.
* The model outputs a probability used to classify mask presence (threshold 0.5).
* You can extend this project to detect faces first before mask classification for improved accuracy.

---

## License

This project is provided as-is for educational purposes.

---

Feel free to contribute or raise issues!

```

---

If you want, I can help you generate a minimal `requirements.txt` or improve the web app further!
```
