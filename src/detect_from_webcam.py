# src/detect_from_webcam.py
import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('models/mask_detector.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    face = cv2.resize(frame, (224, 224)) / 255.0
    face = np.expand_dims(face, axis=0)

    pred = model.predict(face)[0][0]
    label = "Mask" if pred < 0.5 else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.imshow("Face Mask Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

