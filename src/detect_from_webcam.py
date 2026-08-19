# src/detect_from_webcam.py
# This script uses a computer's webcam to continuously capture video,
# send each video frame through a trained TensorFlow mask-detection model,
# and display "Mask" or "No Mask" on the webcam window.


# Import OpenCV.
# OpenCV is used to:
# - Access the webcam
# - Capture video frames
# - Resize images
# - Draw text on frames
# - Display the webcam window
# - Detect keyboard input
import cv2


# Import TensorFlow.
# TensorFlow is used to load the trained mask-detection model
# and make predictions.
import tensorflow as tf


# Import NumPy.
# NumPy is used to manipulate the image arrays
# and add the batch dimension required by the model.
import numpy as np


# ---------------------------------------------------------
# LOAD THE TRAINED MODEL
# ---------------------------------------------------------


# Load the trained TensorFlow/Keras model.
#
# The model is stored in:
#
#     models/mask_detector.h5
#
# The .h5 file contains the trained neural-network model
# and its learned weights.
model = tf.keras.models.load_model(
    'models/mask_detector.h5'
)


# ---------------------------------------------------------
# OPEN THE WEBCAM
# ---------------------------------------------------------


# Create a VideoCapture object.
#
# VideoCapture(0) tells OpenCV to use the computer's
# default/first webcam.
#
# Common camera indexes:
#
#     0 → first/default webcam
#     1 → second webcam
#     2 → third webcam
#
# cap will be used to continuously retrieve frames
# from the webcam.
cap = cv2.VideoCapture(0)


# ---------------------------------------------------------
# CONTINUOUSLY READ WEBCAM FRAMES
# ---------------------------------------------------------


# Start an infinite loop.
#
# The webcam needs to continuously capture frames,
# so we keep running this loop until the user presses 'q'
# or the webcam fails.
while True:


    # Read one frame from the webcam.
    #
    # cap.read() returns TWO values:
    #
    # ret   → tells us whether the frame was successfully captured
    #
    # frame → contains the actual image/frame
    #
    # Example:
    #
    # ret = True
    # frame = image data
    ret, frame = cap.read()


    # Check whether the webcam successfully captured a frame.
    #
    # If ret is False, something went wrong with the camera,
    # so we stop the loop.
    #
    # This is equivalent to:
    #
    # if not ret:
    #     break
    if not ret:
        break


    # -----------------------------------------------------
    # PREPROCESS THE WEBCAM FRAME
    # -----------------------------------------------------


    # Resize the webcam frame to 224 x 224 pixels.
    #
    # The trained neural network expects an image
    # with this size.
    #
    # / 255.0 normalizes the pixel values.
    #
    # Original pixel values:
    #
    #     0 - 255
    #
    # After dividing by 255:
    #
    #     0.0 - 1.0
    face = cv2.resize(
        frame,
        (224, 224)
    ) / 255.0


    # Add a batch dimension to the image.
    #
    # Before expand_dims:
    #
    #     (224, 224, 3)
    #
    # After expand_dims:
    #
    #     (1, 224, 224, 3)
    #
    # The "1" means that we are sending
    # one image to the model.
    face = np.expand_dims(
        face,
        axis=0
    )


    # -----------------------------------------------------
    # MAKE THE PREDICTION
    # -----------------------------------------------------


    # Send the processed webcam frame through the model.
    #
    # model.predict(face) returns the model's prediction.
    #
    # [0] gets the prediction for the first image.
    #
    # [0] gets the actual prediction value.
    #
    # For example:
    #
    # model.predict(face)
    #
    # might return:
    #
    # [[0.23]]
    #
    # Then:
    #
    # [0] → [0.23]
    #
    # and:
    #
    # [0] → 0.23
    #
    # Therefore:
    #
    # pred = 0.23
    pred = model.predict(face)[0][0]


    # Determine whether the model predicts a mask.
    #
    # If pred is below 0.5:
    #
    #     Mask
    #
    # Otherwise:
    #
    #     No Mask
    #
    # This is a conditional/ternary expression.
    #
    # General form:
    #
    #     value_if_true if condition else value_if_false
    label = "Mask" if pred < 0.5 else "No Mask"


    # Choose the text color based on the prediction.
    #
    # OpenCV uses the color format:
    #
    #     (Blue, Green, Red)
    #
    # NOT:
    #
    #     (Red, Green, Blue)
    #
    # Therefore:
    #
    # (0, 255, 0) → Green
    #
    # (0, 0, 255) → Red
    #
    # If the prediction is "Mask":
    #
    #     color = green
    #
    # Otherwise:
    #
    #     color = red
    color = (
        (0, 255, 0)
        if label == "Mask"
        else (0, 0, 255)
    )


    # -----------------------------------------------------
    # DISPLAY THE PREDICTION ON THE VIDEO
    # -----------------------------------------------------


    # Draw the prediction text directly onto the webcam frame.
    #
    # cv2.putText() has several arguments:
    #
    # frame
    #     → the image we want to modify
    #
    # label
    #     → the text we want to display
    #
    # (20, 50)
    #     → position of the text
    #        x = 20
    #        y = 50
    #
    # cv2.FONT_HERSHEY_SIMPLEX
    #     → font used for the text
    #
    # 1.2
    #     → font size
    #
    # color
    #     → text color
    #
    # 2
    #     → thickness of the text
    cv2.putText(
        frame,
        label,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        2
    )


    # Display the webcam frame in a window.
    #
    # "Face Mask Detector" is the title of the window.
    #
    # frame contains:
    #
    #     Webcam video
    #     +
    #     Mask/No Mask text
    cv2.imshow(
        "Face Mask Detector",
        frame
    )


    # Wait for a keyboard key for 1 millisecond.
    #
    # cv2.waitKey(1) waits approximately 1 millisecond
    # for a keyboard input.
    #
    # & 0xFF ensures that we only keep the relevant
    # lower 8 bits of the keyboard value.
    #
    # ord('q') converts the character 'q'
    # into its numerical keyboard code.
    #
    # Therefore:
    #
    # if the user presses q:
    #
    #     cv2.waitKey(1) & 0xFF == ord('q')
    #
    # becomes True.
    #
    # Then break exits the while loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ---------------------------------------------------------
# CLEAN UP
# ---------------------------------------------------------


# Release the webcam.
#
# This tells the operating system that our program
# is finished using the camera.
#
# This is important because otherwise another application
# might not be able to access the webcam.
cap.release()


# Close all OpenCV windows.
#
# This removes the "Face Mask Detector" window
# and any other OpenCV windows created by the program.
cv2.destroyAllWindows()
