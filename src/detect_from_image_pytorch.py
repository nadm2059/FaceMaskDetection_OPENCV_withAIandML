# Import the sys module.
# sys allows the program to access command-line arguments.
import sys


# Import OpenCV.
# OpenCV is used to read the image from the file path.
import cv2


# Import PyTorch.
# PyTorch is the deep-learning framework used to load and run the model.
import torch


# Import torchvision's image transformation tools.
# These tools are used to preprocess the image before giving it to the model.
import torchvision.transforms as transforms


# Import the MobileNetV2 neural-network architecture.
from torchvision.models import mobilenet_v2


# Import NumPy.
# NumPy can be used for numerical operations and working with arrays.
# Note: This particular script does not actually use np later,
# so this import could be removed without affecting the program.
import numpy as np


# Determine which device should be used to run the neural network.
#
# torch.cuda.is_available() checks whether a CUDA-compatible NVIDIA GPU
# is available.
#
# If a GPU is available:
#
#     DEVICE = "cuda"
#
# Otherwise:
#
#     DEVICE = "cpu"
#
# Using a GPU can make neural-network predictions much faster.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# LOAD MODEL ARCHITECTURE AND WEIGHTS
# ---------------------------------------------------------


# Create a MobileNetV2 model.
#
# pretrained=False means:
#
#     "Do not automatically load pretrained ImageNet weights."
#
# We are going to load our own trained mask-detection weights
# from the .pth file later.
model = mobilenet_v2(pretrained=False)


# Replace MobileNetV2's original classifier with a new classifier
# designed for our mask-detection problem.
#
# torch.nn.Sequential() creates a sequence of neural-network layers.
model.classifier = torch.nn.Sequential(

    # Dropout randomly disables 20% of neurons during training.
    #
    # Dropout is useful for reducing overfitting.
    #
    # During model.eval(), Dropout is disabled.
    torch.nn.Dropout(0.2),

    # Create a fully connected (linear) layer.
    #
    # model.last_channel contains the number of features
    # produced by MobileNetV2.
    #
    # The output size is 1 because we are doing binary classification:
    #
    #     Mask
    #     No Mask
    torch.nn.Linear(model.last_channel, 1)
)


# Load the trained weights from the saved PyTorch file.
#
# 'models/mask_detector_pytorch.pth'
# is the location of the trained model weights.
#
# torch.load() reads the saved weights from the file.
#
# map_location=DEVICE makes sure the weights are loaded onto
# the correct device, either CPU or GPU.
#
# load_state_dict() places those weights into our model architecture.
model.load_state_dict(
    torch.load(
        'models/mask_detector_pytorch.pth',
        map_location=DEVICE
    )
)


# Move the entire model to the selected device.
#
# If DEVICE is "cuda", the model is moved to the GPU.
#
# If DEVICE is "cpu", the model remains on the CPU.
model.to(DEVICE)


# Put the model into evaluation mode.
#
# This tells PyTorch that we are using the model for prediction,
# not training.
#
# This is important because layers such as Dropout behave differently
# during training and evaluation.
model.eval()


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------


# Create a sequence of image transformations.
#
# transforms.Compose() means that each transformation will be applied
# in order from top to bottom.
#
# The image will go through:
#
# OpenCV image
#      ↓
# PIL image
#      ↓
# Resize
#      ↓
# PyTorch Tensor
#      ↓
# Normalize
transform = transforms.Compose([


    # Convert the OpenCV/NumPy image into a PIL image.
    #
    # OpenCV gives us a NumPy array.
    #
    # To use torchvision's Resize transformation,
    # we convert the NumPy array into a PIL Image.
    transforms.ToPILImage(),


    # Resize the image to 224 × 224 pixels.
    #
    # MobileNetV2 expects images of this size.
    transforms.Resize((224, 224)),


    # Convert the PIL image into a PyTorch tensor.
    #
    # The pixel values are converted from approximately:
    #
    #     0 - 255
    #
    # to:
    #
    #     0.0 - 1.0
    #
    # The image dimensions are also rearranged into the format
    # expected by PyTorch:
    #
    #     [channels, height, width]
    #
    # For an RGB image:
    #
    #     [3, 224, 224]
    transforms.ToTensor(),


    # Normalize the image using the mean and standard deviation
    # expected by the pretrained MobileNetV2/ImageNet preprocessing.
    #
    # The first list contains the mean values for:
    #
    #     Red   = 0.485
    #     Green = 0.456
    #     Blue  = 0.406
    #
    # The second list contains the standard deviations for:
    #
    #     Red   = 0.229
    #     Green = 0.224
    #     Blue  = 0.225
    #
    # Normalization helps put the input data into the range/distribution
    # expected by the neural network.
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# READ INPUT IMAGE
# ---------------------------------------------------------


# Get the image path from the command-line arguments.
#
# sys.argv is a list containing arguments passed to the program.
#
# For example:
#
#     python detect_pytorch.py test.jpg
#
# sys.argv[0] = "detect_pytorch.py"
#
# sys.argv[1] = "test.jpg"
#
# Therefore, img_path will contain:
#
#     "test.jpg"
img_path = sys.argv[1]


# Read the image from the specified file path using OpenCV.
#
# cv2.imread() loads the image as a NumPy array.
#
# If the image is successfully loaded:
#
#     img contains the image data.
#
# If OpenCV cannot read the image:
#
#     img will be None.
img = cv2.imread(img_path)


# Check whether OpenCV successfully loaded the image.
#
# "img is None" means that the image could not be read.
#
# This could happen if:
#
# - The file does not exist.
# - The path is incorrect.
# - The file is corrupted.
# - OpenCV does not support the file format.
if img is None:


    # Print an error message telling the user that
    # the image could not be read.
    #
    # The f-string inserts the image path into the message.
    print(
        f"Error: Unable to read image {img_path}"
    )


    # Stop the program with an error/exit status of 1.
    #
    # sys.exit(0) usually represents successful termination.
    #
    # sys.exit(1) represents an error.
    sys.exit(1)


# ---------------------------------------------------------
# PREPARE IMAGE FOR THE MODEL
# ---------------------------------------------------------


# Apply all of the preprocessing transformations to the image.
#
# transform(img) performs:
#
#     NumPy/OpenCV image
#            ↓
#        PIL Image
#            ↓
#      224 × 224 resize
#            ↓
#       PyTorch Tensor
#            ↓
#        Normalization
#
# The resulting tensor initially has the shape:
#
#     [3, 224, 224]
#
# .unsqueeze(0) adds a batch dimension:
#
#     [1, 3, 224, 224]
#
# The "1" means we are sending one image to the model.
#
# .to(DEVICE) moves the tensor to the same device as the model.
#
# For example:
#
#     CPU → CPU
#
# or:
#
#     GPU → GPU
input_tensor = transform(img).unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------
# MAKE PREDICTION
# ---------------------------------------------------------


# Disable gradient calculations while making the prediction.
#
# Gradients are needed when training a neural network,
# but they are not needed when making predictions.
#
# torch.no_grad():
#
# - Reduces memory usage.
# - Makes inference more efficient.
# - Prevents PyTorch from building a computation graph.
with torch.no_grad():


    # Pass the preprocessed image into the neural network.
    #
    # input_tensor has a shape similar to:
    #
    #     [1, 3, 224, 224]
    #
    # The model processes the image and returns its raw output.
    output = model(input_tensor)


    # Convert the model's raw output into a probability between 0 and 1.
    #
    # A neural network with a single output neuron commonly produces
    # a raw value called a "logit."
    #
    # torch.sigmoid() converts the logit into a value between:
    #
    #     0 and 1
    #
    # .item() converts the single PyTorch tensor value
    # into a regular Python number.
    prob = torch.sigmoid(output).item()


# ---------------------------------------------------------
# DETERMINE THE FINAL CLASS
# ---------------------------------------------------------


# Use 0.5 as the classification threshold.
#
# If:
#
#     prob < 0.5
#
# the program considers the image to contain a mask.
#
# Otherwise:
#
#     prob >= 0.5
#
# the program considers the image to contain no mask.
#
# This is a conditional/ternary expression.
#
# General form:
#
#     value_if_true if condition else value_if_false
label = "Mask" if prob < 0.5 else "No Mask"


# Print the final prediction to the terminal.
#
# {label} inserts the predicted label.
#
# {prob:.4f} formats the probability to exactly
# four digits after the decimal point.
#
# For example:
#
#     Prediction: Mask (probability=0.2734)
#
# or:
#
#     Prediction: No Mask (probability=0.8231)
print(
    f"Prediction: {label} (probability={prob:.4f})"
)
