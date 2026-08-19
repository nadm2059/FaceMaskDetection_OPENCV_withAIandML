# Import OpenCV.
# OpenCV is used to:
# - Access the webcam
# - Capture video frames
# - Draw text on the video
# - Display the webcam window
# - Detect keyboard input
import cv2


# Import PyTorch.
# PyTorch is used to load the neural-network model,
# move it to the CPU/GPU, and make predictions.
import torch


# Import torchvision's image transformation tools.
# These transformations prepare webcam images
# so they have the same format expected by the model.
import torchvision.transforms as transforms


# Import the MobileNetV2 neural-network architecture.
from torchvision.models import mobilenet_v2


# ---------------------------------------------------------
# SELECT CPU OR GPU
# ---------------------------------------------------------


# Check whether a CUDA-compatible NVIDIA GPU is available.
#
# torch.cuda.is_available()
#     → returns True if CUDA/GPU is available
#     → returns False otherwise
#
# If CUDA is available:
#
#     DEVICE = "cuda"
#
# Otherwise:
#
#     DEVICE = "cpu"
#
# The device determines where the neural network
# and input images will be processed.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# LOAD MODEL ARCHITECTURE AND WEIGHTS
# ---------------------------------------------------------


# Create a MobileNetV2 model architecture.
#
# pretrained=False means that we are NOT automatically
# downloading/loading ImageNet pretrained weights.
#
# We will load our own trained mask-detection weights
# from the .pth file below.
model = mobilenet_v2(pretrained=False)


# Replace MobileNetV2's original classifier.
#
# MobileNetV2 was originally designed to classify
# 1,000 ImageNet categories.
#
# We don't need those 1,000 classes.
#
# We need a binary classifier:
#
#     Mask
#     No Mask
#
# Therefore, we replace the original classifier.
model.classifier = torch.nn.Sequential(


    # Dropout layer.
    #
    # 0.2 means 20% dropout.
    #
    # During training, Dropout randomly turns off
    # 20% of neurons to help reduce overfitting.
    #
    # During model.eval(), Dropout is disabled.
    torch.nn.Dropout(0.2),


    # Fully connected/linear layer.
    #
    # model.last_channel represents the number of features
    # produced by MobileNetV2.
    #
    # The output size is 1 because we have one binary
    # classification output.
    torch.nn.Linear(model.last_channel, 1)
)


# Load the trained weights from the .pth file.
#
# torch.load() reads the saved model weights.
#
# map_location=DEVICE makes sure the weights are loaded
# onto the correct device.
#
# load_state_dict() puts the saved weights into
# our MobileNetV2 model.
model.load_state_dict(
    torch.load(
        'models/mask_detector_pytorch.pth',
        map_location=DEVICE
    )
)


# Move the model to the selected device.
#
# If DEVICE is:
#
#     "cuda" → model goes to the GPU
#
#     "cpu"  → model stays on the CPU
model.to(DEVICE)


# Put the model into evaluation mode.
#
# This tells PyTorch:
#
#     "We are making predictions, not training."
#
# This is important because layers such as Dropout
# behave differently during training and evaluation.
model.eval()


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------


# Create a preprocessing pipeline.
#
# transforms.Compose() means:
#
#     Apply each transformation in order.
#
# The pipeline is:
#
#     OpenCV image
#          ↓
#     PIL image
#          ↓
#     Resize
#          ↓
#     Tensor
#          ↓
#     Normalize
transform = transforms.Compose([


    # Convert the OpenCV/NumPy image into a PIL image.
    #
    # OpenCV gives us the webcam frame as a NumPy array.
    #
    # ToPILImage() converts that array into
    # a PIL Image object.
    transforms.ToPILImage(),


    # Resize the image to 224 × 224 pixels.
    #
    # MobileNetV2 expects images with this spatial size.
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
    # The dimensions become:
    #
    #     [channels, height, width]
    #
    # For an RGB image:
    #
    #     [3, 224, 224]
    transforms.ToTensor(),


    # Normalize the image.
    #
    # These values are the ImageNet mean and standard deviation.
    #
    # Mean:
    #
    #     Red   = 0.485
    #     Green = 0.456
    #     Blue  = 0.406
    #
    # Standard deviation:
    #
    #     Red   = 0.229
    #     Green = 0.224
    #     Blue  = 0.225
    #
    # Normalization makes the input distribution similar
    # to what MobileNetV2 expects.
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# OPEN THE WEBCAM
# ---------------------------------------------------------


# Open the computer's first/default webcam.
#
# 0 means the first camera.
#
# If you had multiple cameras, you might use:
#
#     0 → first camera
#     1 → second camera
#     2 → third camera
cap = cv2.VideoCapture(0)


# ---------------------------------------------------------
# CONTINUOUSLY PROCESS VIDEO
# ---------------------------------------------------------


# Start an infinite loop.
#
# The loop continuously captures webcam frames
# and sends them through the neural network.
#
# It will continue until:
#
#     1. The webcam fails
#     OR
#     2. The user presses 'q'
while True:


    # Capture one frame from the webcam.
    #
    # cap.read() returns two values:
    #
    # ret
    #     → True if the frame was successfully captured
    #     → False if there was an error
    #
    # frame
    #     → the actual webcam image
    ret, frame = cap.read()


    # Check whether the frame was successfully captured.
    #
    # If ret is False, stop the loop.
    if not ret:
        break


    # -----------------------------------------------------
    # PREPROCESS THE CURRENT FRAME
    # -----------------------------------------------------


    # Apply the preprocessing pipeline to the webcam frame.
    #
    # transform(frame) performs:
    #
    #     1. NumPy/OpenCV image
    #            ↓
    #     2. PIL Image
    #            ↓
    #     3. Resize to 224 × 224
    #            ↓
    #     4. Convert to PyTorch Tensor
    #            ↓
    #     5. Normalize
    #
    # After transform(frame), the tensor has approximately:
    #
    #     [3, 224, 224]
    #
    # .unsqueeze(0) adds a batch dimension:
    #
    #     [1, 3, 224, 224]
    #
    # The "1" means one image.
    #
    # .to(DEVICE) moves the tensor to the same device
    # as the model.
    input_tensor = transform(frame).unsqueeze(0).to(DEVICE)


    # -----------------------------------------------------
    # MAKE PREDICTION
    # -----------------------------------------------------


    # Tell PyTorch that we do not need gradients.
    #
    # Gradients are required during training,
    # but they are unnecessary during prediction.
    #
    # Using no_grad():
    #
    # - Reduces memory usage
    # - Makes inference more efficient
    # - Prevents unnecessary gradient calculations
    with torch.no_grad():


        # Send the current webcam frame into the neural network.
        #
        # The model receives:
        #
        #     [1, 3, 224, 224]
        #
        # and returns a raw prediction called a logit.
        output = model(input_tensor)


        # Convert the raw model output into a probability.
        #
        # torch.sigmoid() converts the raw output into
        # a value between 0 and 1.
        #
        # .item() converts the single PyTorch tensor value
        # into a normal Python number.
        prob = torch.sigmoid(output).item()


    # -----------------------------------------------------
    # CLASSIFY THE FRAME
    # -----------------------------------------------------


    # Use 0.5 as the classification threshold.
    #
    # If:
    #
    #     prob < 0.5
    #
    # then the program predicts:
    #
    #     Mask
    #
    # Otherwise:
    #
    #     No Mask
    #
    # This is a conditional/ternary expression.
    label = "Mask" if prob < 0.5 else "No Mask"


    # Choose the text color based on the prediction.
    #
    # OpenCV uses BGR color order:
    #
    #     (Blue, Green, Red)
    #
    # Green:
    #
    #     (0, 255, 0)
    #
    # Red:
    #
    #     (0, 0, 255)
    #
    # Therefore:
    #
    # Mask    → green
    # No Mask → red
    color = (
        (0, 255, 0)
        if label == "Mask"
        else (0, 0, 255)
    )


    # -----------------------------------------------------
    # DISPLAY PREDICTION ON VIDEO
    # -----------------------------------------------------


    # Draw the prediction and probability on the webcam frame.
    #
    # f"{label} ({prob:.2f})" creates text such as:
    #
    #     Mask (0.23)
    #
    # or:
    #
    #     No Mask (0.81)
    #
    # {prob:.2f} means:
    #
    #     Format prob with 2 digits after the decimal point.
    #
    # (20, 50) specifies where the text appears:
    #
    #     x = 20
    #     y = 50
    #
    # cv2.FONT_HERSHEY_SIMPLEX specifies the font.
    #
    # 1.2 specifies the font size.
    #
    # color specifies the text color.
    #
    # 2 specifies the thickness of the text.
    cv2.putText(
        frame,
        f"{label} ({prob:.2f})",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        2
    )


    # Display the current webcam frame.
    #
    # The window title will be:
    #
    #     Face Mask Detector (PyTorch)
    #
    # The frame contains both:
    #
    #     Webcam video
    #
    # and:
    #
    #     Prediction + probability
    cv2.imshow(
        "Face Mask Detector (PyTorch)",
        frame
    )


    # Wait approximately 1 millisecond for a keyboard input.
    #
    # cv2.waitKey(1)
    #     → checks whether a key was pressed
    #
    # & 0xFF
    #     → keeps the relevant 8 bits of the key value
    #
    # ord('q')
    #     → converts the character 'q' into its numerical code
    #
    # Therefore, this condition checks:
    #
    #     "Did the user press q?"
    if cv2.waitKey(1) & 0xFF == ord('q'):


        # Exit the while loop.
        #
        # The program will then continue to the cleanup code.
        break


# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------


# Release the webcam.
#
# This tells the operating system that the program
# is finished using the camera.
cap.release()


# Close all OpenCV windows.
#
# This closes the Face Mask Detector window.
cv2.destroyAllWindows()
