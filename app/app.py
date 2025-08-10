# app/app.py

import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Streamlit app title
st.title("😷 Face Mask Detection App")

# Backend selection
backend = st.radio("Choose model backend:", ["TensorFlow", "PyTorch"])

# Load models once (cache for performance)
@st.cache(allow_output_mutation=True)
def load_tf_model():
    import tensorflow as tf
    return tf.keras.models.load_model('models/mask_detector.h5')

@st.cache(allow_output_mutation=True)
def load_pytorch_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use MobileNet_V2_Weights directly, not as attribute of mobilenet_v2
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(model.last_channel, 1)
    )
    model.load_state_dict(torch.load('models/mask_detector_pytorch.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device


# Image uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if backend == "TensorFlow":
        model = load_tf_model()

        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

        prediction = model.predict(img_array)

        if prediction.shape[1] == 1:
            label = "✅ Mask Detected" if prediction[0][0] < 0.5 else "❌ No Mask Detected"
        else:
            class_names = ['with_mask', 'without_mask']
            predicted_class = np.argmax(prediction[0])
            label = f"🧾 Prediction: {class_names[predicted_class]}"

        st.subheader(label)

    else:  # PyTorch backend
        model, device = load_pytorch_model()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()

        label = "✅ Mask Detected" if prob < 0.5 else "❌ No Mask Detected"
        st.subheader(f"{label} (probability={prob:.4f})")


