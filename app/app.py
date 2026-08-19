# app/app.py
# This is the Python file that contains the Streamlit application.

# Import Streamlit, which is used to create the web application interface.
import streamlit as st

# Import the Image class from Pillow (PIL) to open and manipulate images.
from PIL import Image

# Import NumPy, which is used to convert images into numerical arrays.
import numpy as np

# Import PyTorch, which is used for loading and running the PyTorch model.
import torch

# Import torchvision's image transformation utilities.
# These are used to resize, convert, and normalize images before prediction.
import torchvision.transforms as transforms

# Import the MobileNetV2 model and the pretrained MobileNetV2 weights.
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# Display the title of the Streamlit application.
# The emoji and text will appear at the top of the webpage.
st.title("😷 Face Mask Detection App")


# Create a radio-button selection that allows the user to choose the model backend.
# "backend" stores the user's selection.
# The two available choices are TensorFlow and PyTorch.
backend = st.radio("Choose model backend:", ["TensorFlow", "PyTorch"])


# Tell Streamlit to cache the result of this function.
# Caching prevents the TensorFlow model from being loaded from disk every time
# Streamlit reruns the script.
#
# allow_output_mutation=True allows the cached model object to be reused
# even though the object itself is mutable.
@st.cache(allow_output_mutation=True)

# Define a function that loads the TensorFlow mask-detection model.
def load_tf_model():

    # Import TensorFlow inside the function.
    # This means TensorFlow is only imported when this function is actually called.
    import tensorflow as tf

    # Load the previously trained Keras/TensorFlow model from the specified file.
    # The .h5 file contains the model architecture and trained weights.
    return tf.keras.models.load_model('models/mask_detector.h5')


# Cache the PyTorch model so that it does not need to be loaded repeatedly
# every time Streamlit reruns the application.
@st.cache(allow_output_mutation=True)

# Define a function that loads and prepares the PyTorch model.
def load_pytorch_model():

    # Check whether a CUDA-compatible NVIDIA GPU is available.
    # If CUDA is available, the model will run on the GPU.
    # Otherwise, it will run on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select the pretrained ImageNet weights for MobileNetV2.
    # These weights are used to initialize the MobileNetV2 model.
    weights = MobileNet_V2_Weights.IMAGENET1K_V1

    # Create a MobileNetV2 model using the selected pretrained weights.
    model = mobilenet_v2(weights=weights)

    # Loop through every parameter in the model.
    for param in model.parameters():

        # Disable gradient calculation for these parameters.
        # This freezes the pretrained MobileNetV2 layers.
        # We are using MobileNetV2 as a feature extractor.
        param.requires_grad = False

    # Replace MobileNetV2's original classifier with a new classifier
    # designed for our binary mask-detection problem.
    model.classifier = torch.nn.Sequential(

        # Add dropout with a probability of 0.2.
        # During training, this randomly disables 20% of the neurons
        # to help reduce overfitting.
        torch.nn.Dropout(0.2),

        # Create a fully connected layer.
        #
        # model.last_channel is the number of features produced by MobileNetV2.
        #
        # The output size is 1 because this is binary classification:
        # mask vs. no mask.
        torch.nn.Linear(model.last_channel, 1)
    )

    # Load the trained mask-detection weights from the .pth file.
    #
    # map_location=device makes sure the model is loaded onto the correct
    # device, either the GPU or CPU.
    model.load_state_dict(
        torch.load(
            'models/mask_detector_pytorch.pth',
            map_location=device
        )
    )

    # Move the entire model to the selected device.
    # This makes the model run on either the GPU or CPU.
    model.to(device)

    # Put the model into evaluation mode.
    #
    # This tells PyTorch that we are using the model for prediction,
    # not training.
    #
    # It also changes the behavior of layers such as Dropout and BatchNorm.
    model.eval()

    # Return both the loaded model and the device being used.
    return model, device


# Create a file-upload button in the Streamlit application.
#
# The user can upload an image.
# The accepted file extensions are JPG, JPEG, and PNG.
uploaded_file = st.file_uploader(
    "Upload an image...",
    type=["jpg", "jpeg", "png"]
)


# Check whether the user actually uploaded a file.
#
# If uploaded_file contains a file, this condition is True.
# If no file was uploaded, it is None and this block is skipped.
if uploaded_file:

    # Open the uploaded image using Pillow.
    #
    # convert("RGB") ensures the image has exactly three color channels:
    # Red, Green, and Blue.
    img = Image.open(uploaded_file).convert("RGB")

    # Display the uploaded image in the Streamlit webpage.
    #
    # caption provides text underneath the image.
    # use_column_width=True makes the image fit the available column width.
    st.image(
        img,
        caption="Uploaded Image",
        use_column_width=True
    )


    # Check which model backend the user selected.
    #
    # If the radio button selection is "TensorFlow",
    # the TensorFlow model will be used.
    if backend == "TensorFlow":

        # Load the TensorFlow mask-detection model.
        #
        # Because the function is cached, the model will not need
        # to be loaded from disk every time.
        model = load_tf_model()


        # Resize the uploaded image to 224 x 224 pixels.
        #
        # MobileNetV2 expects images of this size.
        img_resized = img.resize((224, 224))


        # Convert the PIL image into a NumPy array.
        #
        # The result contains numerical pixel values.
        #
        # / 255.0 changes pixel values from:
        #
        #     0 - 255
        #
        # to:
        #
        #     0.0 - 1.0
        #
        # This is a common normalization method for neural networks.
        img_array = np.array(img_resized) / 255.0


        # Add an extra dimension representing the batch size.
        #
        # Before expand_dims:
        #
        #     (224, 224, 3)
        #
        # After expand_dims:
        #
        #     (1, 224, 224, 3)
        #
        # The "1" means we are predicting one image.
        img_array = np.expand_dims(img_array, axis=0)


        # Pass the processed image through the TensorFlow model.
        #
        # The model returns its prediction.
        prediction = model.predict(img_array)


        # Check whether the model's output has one value per image.
        #
        # prediction.shape[1] == 1 means the model is producing
        # a single output value for binary classification.
        if prediction.shape[1] == 1:

            # Check the predicted probability.
            #
            # prediction[0][0] gets the first prediction value.
            #
            # If it is below 0.5, we interpret it as "Mask Detected".
            #
            # Otherwise, we interpret it as "No Mask Detected".
            label = (
                "✅ Mask Detected"
                if prediction[0][0] < 0.5
                else "❌ No Mask Detected"
            )


        # If the model does not produce one output value,
        # assume that it produces multiple class probabilities.
        else:

            # Define the names corresponding to the model's output classes.
            #
            # Index 0 = with_mask
            # Index 1 = without_mask
            class_names = ['with_mask', 'without_mask']


            # Find the index of the class with the largest prediction value.
            #
            # np.argmax() returns the index containing the maximum value.
            #
            # For example:
            #
            # [0.8, 0.2] -> index 0 -> with_mask
            #
            # [0.1, 0.9] -> index 1 -> without_mask
            predicted_class = np.argmax(prediction[0])


            # Create a label using the predicted class.
            #
            # f"" allows us to insert the class name into the string.
            label = f"🧾 Prediction: {class_names[predicted_class]}"


        # Display the prediction as a subheading in the Streamlit application.
        st.subheader(label)


    # If the user did not choose TensorFlow,
    # the PyTorch backend is used instead.
    else:

        # Load the PyTorch model and determine whether it uses
        # the CPU or GPU.
        #
        # model = the neural network
        # device = CPU or CUDA GPU
        model, device = load_pytorch_model()


        # Create a sequence of image transformations.
        #
        # transforms.Compose() applies the transformations
        # one after another.
        transform = transforms.Compose([

            # Resize the image to 224 x 224 pixels.
            # MobileNetV2 expects this image size.
            transforms.Resize((224, 224)),

            # Convert the PIL image into a PyTorch tensor.
            #
            # It also changes the pixel range from approximately:
            #
            #     0 - 255
            #
            # to:
            #
            #     0.0 - 1.0
            transforms.ToTensor(),

            # Normalize the image using the mean and standard deviation
            # used by the pretrained ImageNet MobileNetV2 model.
            #
            # The three values correspond to:
            #
            # Red
            # Green
            # Blue
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])


        # Apply all of the transformations to the uploaded image.
        #
        # transform(img) converts the image into a PyTorch tensor.
        #
        # unsqueeze(0) adds the batch dimension.
        #
        # Before unsqueeze:
        #
        #     [3, 224, 224]
        #
        # After unsqueeze:
        #
        #     [1, 3, 224, 224]
        #
        # .to(device) moves the tensor to the CPU or GPU.
        input_tensor = transform(img).unsqueeze(0).to(device)


        # Tell PyTorch that we do not need to calculate gradients.
        #
        # Gradients are necessary during training, but not during prediction.
        #
        # This reduces memory usage and makes inference faster.
        with torch.no_grad():

            # Pass the image tensor through the PyTorch model.
            #
            # The model returns its raw output, often called a logit.
            output = model(input_tensor)


            # Convert the model's raw output into a probability
            # between 0 and 1 using the sigmoid function.
            #
            # .item() converts the single PyTorch tensor value
            # into a regular Python number.
            prob = torch.sigmoid(output).item()


        # Determine whether a mask is detected.
        #
        # If the probability is below 0.5:
        #
        #     Mask Detected
        #
        # Otherwise:
        #
        #     No Mask Detected
        label = (
            "✅ Mask Detected"
            if prob < 0.5
            else "❌ No Mask Detected"
        )


        # Display the prediction and probability.
        #
        # f"{prob:.4f}" formats the probability to exactly
        # four digits after the decimal point.
        #
        # Example:
        #
        # probability=0.2734
        st.subheader(
            f"{label} (probability={prob:.4f})"
        )
