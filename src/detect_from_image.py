# src/detect_from_image.py
import cv2
import tensorflow as tf
import numpy as np
import sys

model = tf.keras.models.load_model('models/mask_detector.h5')

img_path = sys.argv[1]
img = cv2.imread(img_path)
resized = cv2.resize(img, (224, 224)) / 255.0
resized = np.expand_dims(resized, axis=0)

pred = model.predict(resized)[0][0]
label = "Mask" if pred < 0.5 else "No Mask"
print(f"Prediction: {label}")
