# app/app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('models/mask_detector.h5')

# Set Streamlit title
st.title("😷 Face Mask Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and preprocess image
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # Predict
    prediction = model.predict(img_array)

    # Interpret prediction (adjust based on your model output)
    if prediction.shape[1] == 1:
        # Binary classification with sigmoid
        label = "✅ Mask Detected" if prediction[0][0] < 0.5 else "❌ No Mask Detected"
    else:
        # Categorical classification with softmax
        class_names = ['with_mask', 'without_mask']  # adjust if needed
        predicted_class = np.argmax(prediction[0])
        label = f"🧾 Prediction: {class_names[predicted_class]}"

    # Show result
    st.subheader(label)


