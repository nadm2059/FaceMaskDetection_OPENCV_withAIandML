# Import Python's built-in os module.
# os is used later to create the "models" directory
# where the trained model will be saved.
import os


# Import the main PyTorch library.
# PyTorch is used to create, train, and run the neural network.
import torch


# Import PyTorch's neural-network tools as "nn"
# and optimization tools as "optim".
#
# nn contains things like:
#     Linear
#     Dropout
#     BCEWithLogitsLoss
#
# optim contains optimizers such as:
#     Adam
#     SGD
from torch import nn, optim


# Import torchvision tools.
#
# datasets:
#     Provides datasets such as ImageFolder.
#
# transforms:
#     Provides image preprocessing and augmentation.
#
# models:
#     Provides pretrained neural-network architectures
#     such as MobileNetV2.
from torchvision import datasets, transforms, models


# Import DataLoader and random_split.
#
# DataLoader:
#     Loads images in batches.
#
# random_split:
#     Splits a dataset into separate portions,
#     such as training and validation.
from torch.utils.data import DataLoader, random_split


# Import PIL's Image class.
#
# PIL is a Python image-processing library.
# It is used here to convert images to RGB format.
from PIL import Image


# =========================================================
# CUSTOM IMAGE TRANSFORM
# =========================================================


# Create a custom transformation class.
#
# The purpose of this class is to make sure every image
# is converted to RGB format.
#
# Some images may use formats such as:
#
#     P  → palette mode
#     RGBA → Red, Green, Blue, Alpha
#     grayscale
#
# MobileNetV2 expects normal 3-channel RGB images.
class ConvertToRGB(object):


    # __call__ allows an object of this class to be used
    # like a function.
    #
    # For example:
    #
    # converter = ConvertToRGB()
    # converter(image)
    #
    # automatically calls this __call__ method.
    def __call__(self, img):


        # Convert the image to RGB format.
        #
        # RGB contains three color channels:
        #
        #     R → Red
        #     G → Green
        #     B → Blue
        #
        # The converted image is returned.
        return img.convert("RGB")


# =========================================================
# MAIN FUNCTION
# =========================================================


# Define the main function.
#
# Putting the training code inside main()
# keeps the program organized.
def main():


    # -----------------------------------------------------
    # TRAINING SETTINGS
    # -----------------------------------------------------


    # Set the size of the images.
    #
    # Every image will eventually be resized/cropped
    # to 224 × 224 pixels.
    #
    # MobileNetV2 commonly uses this input size.
    IMG_SIZE = 224


    # Number of images processed together in one batch.
    #
    # Here:
    #
    #     32 images
    #
    # are processed before the model's weights are updated.
    BATCH_SIZE = 32


    # Number of complete passes through the training data.
    #
    # 5 epochs means the model sees the training dataset
    # approximately 5 times.
    EPOCHS = 5


    # Location of the image dataset.
    #
    # The directory should contain:
    #
    # data/train/
    # ├── with_mask/
    # │   ├── image1.jpg
    # │   ├── image2.jpg
    # │   └── ...
    # │
    # └── without_mask/
    #     ├── image1.jpg
    #     ├── image2.jpg
    #     └── ...
    DATA_DIR = 'c:/ML_Projects_83/facemaskdetection/data/train'


    # Decide whether the model should use:
    #
    #     CUDA/GPU
    #
    # or:
    #
    #     CPU
    #
    # torch.cuda.is_available()
    # returns True if a CUDA-compatible NVIDIA GPU
    # is available.
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


    # =====================================================
    # IMAGE PREPROCESSING / DATA AUGMENTATION
    # =====================================================


    # Create a sequence of image transformations.
    #
    # transforms.Compose() means:
    #
    #     Apply each transformation in order.
    transform = transforms.Compose([


        # Convert every image to RGB.
        #
        # This ensures that the neural network receives
        # exactly 3 color channels.
        ConvertToRGB(),


        # Randomly crop and resize the image.
        #
        # The final image will be:
        #
        #     224 × 224
        #
        # scale=(0.8, 1.0) means the crop can contain
        # approximately 80% to 100% of the original image
        # area before being resized.
        #
        # This is data augmentation.
        transforms.RandomResizedCrop(
            IMG_SIZE,
            scale=(0.8, 1.0)
        ),


        # Randomly flip the image horizontally.
        #
        # This gives the model additional variations
        # of the training images.
        #
        # Example:
        #
        #     Person facing left
        #            ↓
        #     Person facing right
        transforms.RandomHorizontalFlip(),


        # Convert the image into a PyTorch tensor.
        #
        # The pixel values are changed approximately from:
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
        # or:
        #
        #     [3, 224, 224]
        transforms.ToTensor(),


        # Normalize the image using ImageNet's
        # mean and standard deviation.
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
        # This is appropriate because we are using
        # MobileNetV2 pretrained on ImageNet.
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


    # =====================================================
    # LOAD DATASET
    # =====================================================


    # Create an ImageFolder dataset.
    #
    # ImageFolder automatically:
    #
    #     1. Looks at the subdirectories.
    #     2. Treats each subdirectory as a class.
    #     3. Loads the images.
    #     4. Applies the transform.
    #     5. Assigns numerical labels.
    #
    # For example, it might create:
    #
    #     with_mask     → 0
    #     without_mask  → 1
    #
    # The exact mapping should be checked using:
    #
    #     full_dataset.class_to_idx
    #
    # because the class mapping depends on alphabetical
    # ordering of the folder names.
    full_dataset = datasets.ImageFolder(
        DATA_DIR,
        transform=transform
    )


    # =====================================================
    # TRAIN / VALIDATION SPLIT
    # =====================================================


    # Calculate how many images should be used for training.
    #
    # 80% of the entire dataset is assigned to training.
    #
    # int() converts the result into an integer.
    train_size = int(
        0.8 * len(full_dataset)
    )


    # Calculate the validation dataset size.
    #
    # Instead of calculating 20% directly, we subtract
    # the training size from the total size.
    #
    # This guarantees:
    #
    #     train_size + val_size = total dataset size
    val_size = len(full_dataset) - train_size


    # Randomly divide the dataset into:
    #
    #     train_dataset → 80%
    #
    #     val_dataset → 20%
    #
    # random_split() randomly selects which images
    # go into each group.
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )


    # =====================================================
    # CREATE DATA LOADERS
    # =====================================================


    # Create a DataLoader for the training dataset.
    #
    # DataLoader loads the images in batches.
    #
    # batch_size=32:
    #     Load 32 images at a time.
    #
    # shuffle=True:
    #     Randomize the order of training images
    #     every epoch.
    #
    # num_workers=4:
    #     Use 4 worker processes to load/preprocess
    #     images in parallel.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )


    # Create a DataLoader for validation.
    #
    # shuffle=False means the validation images
    # don't need to be randomly reordered.
    #
    # We still process them in batches of 32.
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )


    # =====================================================
    # LOAD PRETRAINED MOBILENETV2
    # =====================================================


    # Select the pretrained MobileNetV2 weights.
    #
    # IMAGENET1K_V1 represents a specific pretrained
    # MobileNetV2 weight version trained on ImageNet.
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1


    # Create MobileNetV2 using the selected pretrained weights.
    #
    # MobileNetV2 has already learned many useful
    # visual features from ImageNet.
    #
    # We will use those features to help detect masks.
    model = models.mobilenet_v2(
        weights=weights
    )


    # =====================================================
    # FREEZE THE BASE MODEL
    # =====================================================


    # Loop through every parameter in MobileNetV2.
    #
    # A parameter is something the neural network can learn,
    # such as a weight or bias.
    for param in model.parameters():


        # Prevent this parameter from being updated
        # during training.
        #
        # False means:
        #
        #     "Don't calculate/update gradients for this."
        #
        # This freezes the pretrained MobileNetV2.
        param.requires_grad = False


    # =====================================================
    # REPLACE THE CLASSIFIER
    # =====================================================


    # Replace MobileNetV2's original classifier.
    #
    # The original MobileNetV2 classifier is designed
    # for ImageNet's 1,000 classes.
    #
    # Our project only needs one binary output.
    model.classifier = nn.Sequential(


        # Dropout randomly disables 20% of neurons
        # during training.
        #
        # This helps reduce overfitting.
        nn.Dropout(0.2),


        # Create a fully connected/linear layer.
        #
        # model.last_channel is the number of features
        # coming from MobileNetV2.
        #
        # 1 means we want one output.
        #
        # The output will be a raw score called a "logit".
        nn.Linear(
            model.last_channel,
            1
        ),
    )


    # Move the entire model to the selected device.
    #
    # If DEVICE == cuda:
    #
    #     model → GPU
    #
    # If DEVICE == cpu:
    #
    #     model → CPU
    model = model.to(DEVICE)


    # =====================================================
    # LOSS FUNCTION
    # =====================================================


    # Create the loss function.
    #
    # BCE stands for:
    #
    #     Binary Cross Entropy
    #
    # "WithLogits" means this loss function expects
    # raw logits from the model.
    #
    # We do NOT put a sigmoid activation directly
    # inside the model because BCEWithLogitsLoss()
    # handles the sigmoid calculation internally
    # in a numerically stable way.
    criterion = nn.BCEWithLogitsLoss()


    # =====================================================
    # OPTIMIZER
    # =====================================================


    # Create the Adam optimizer.
    #
    # The optimizer is responsible for changing
    # the model's trainable parameters based on
    # the calculated gradients.
    #
    # model.classifier.parameters()
    # means ONLY the new classifier parameters
    # are being trained.
    #
    # lr=0.001 is the learning rate.
    #
    # Learning rate controls how large each update
    # to the model's weights should be.
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=0.001
    )


    # =====================================================
    # TRAINING LOOP
    # =====================================================


    # Loop through the number of epochs.
    #
    # If EPOCHS = 5:
    #
    #     epoch = 0
    #     epoch = 1
    #     epoch = 2
    #     epoch = 3
    #     epoch = 4
    #
    # This represents 5 complete passes through
    # the training dataset.
    for epoch in range(EPOCHS):


        # Put the model into training mode.
        #
        # This is important because layers such as
        # Dropout behave differently during training.
        model.train()


        # Variable used to accumulate the total
        # training loss for the epoch.
        running_loss = 0.0


        # Number of correctly classified training images.
        correct = 0


        # Total number of training images processed.
        total = 0


        # Loop through the training batches.
        #
        # Each iteration gives:
        #
        #     inputs → batch of images
        #
        #     labels → corresponding class labels
        for inputs, labels in train_loader:


            # Move the input images to the selected device.
            #
            # Example:
            #
            #     CPU → GPU
            #
            # if CUDA is being used.
            #
            # labels.float()
            # converts the labels to floating-point numbers.
            #
            # .to(DEVICE)
            # moves the labels to CPU/GPU.
            #
            # .unsqueeze(1)
            # changes the label shape.
            #
            # Example:
            #
            #     [32]
            #
            # becomes:
            #
            #     [32, 1]
            #
            # This matches the model's output shape.
            inputs, labels = (
                inputs.to(DEVICE),
                labels.float().to(DEVICE).unsqueeze(1)
            )


            # Clear the gradients from the previous batch.
            #
            # PyTorch accumulates gradients by default,
            # so we need to reset them before each update.
            optimizer.zero_grad()


            # Send the input images through the neural network.
            #
            # Example input shape:
            #
            #     [32, 3, 224, 224]
            #
            # The model returns one raw output for each image.
            #
            # Example:
            #
            #     [32, 1]
            outputs = model(inputs)


            # Calculate the prediction error.
            #
            # outputs:
            #     model's raw logits
            #
            # labels:
            #     actual answers
            #
            # criterion calculates how different
            # the predictions are from the true labels.
            loss = criterion(
                outputs,
                labels
            )


            # Calculate gradients using backpropagation.
            #
            # The model determines how much each trainable
            # parameter contributed to the error.
            loss.backward()


            # Update the trainable model parameters.
            #
            # Adam uses the calculated gradients to change
            # the classifier weights.
            optimizer.step()


            # Add the batch's total loss to running_loss.
            #
            # loss.item()
            # converts the PyTorch tensor containing the loss
            # into a normal Python number.
            #
            # inputs.size(0)
            # gives the number of images in the batch.
            #
            # Multiplying by the batch size lets us later
            # calculate the average loss across the entire epoch.
            running_loss += (
                loss.item() * inputs.size(0)
            )


            # Convert the model's raw logits into probabilities.
            #
            # sigmoid converts:
            #
            #     raw logits
            #
            # into:
            #
            #     values between 0 and 1
            #
            # >= 0.5 creates a binary prediction:
            #
            #     probability >= 0.5 → 1
            #
            #     probability < 0.5 → 0
            #
            # .float() converts True/False values
            # into 1.0/0.0.
            preds = (
                torch.sigmoid(outputs) >= 0.5
            ).float()


            # Compare predictions with the actual labels.
            #
            # (preds == labels)
            # produces True/False values.
            #
            # .sum()
            # counts how many are True.
            #
            # .item()
            # converts the result to a Python number.
            #
            # This gives the number of correct predictions
            # in the current batch.
            correct += (
                (preds == labels).sum().item()
            )


            # Add the number of images in the current batch
            # to the total number of images processed.
            total += labels.size(0)


        # Calculate average training loss.
        #
        # running_loss contains the accumulated loss
        # for all training examples.
        #
        # total is the number of training examples.
        #
        # Therefore:
        #
        #     average loss = total loss / total examples
        train_loss = running_loss / total


        # Calculate training accuracy.
        #
        # Accuracy =
        #
        #     correct predictions
        #     -------------------
        #     total predictions
        train_acc = correct / total


        # =================================================
        # VALIDATION
        # =================================================


        # Put the model into evaluation mode.
        #
        # This tells PyTorch that we are evaluating
        # rather than training.
        #
        # For example, Dropout is disabled.
        model.eval()


        # Store the total validation loss.
        val_loss = 0.0


        # Store the number of correct validation predictions.
        val_correct = 0


        # Store the number of validation examples processed.
        val_total = 0


        # Disable gradient calculations.
        #
        # We don't need gradients while validating.
        #
        # This reduces memory usage and makes validation faster.
        with torch.no_grad():


            # Loop through validation batches.
            for inputs, labels in val_loader:


                # Move validation images to the selected device.
                #
                # Convert labels to floating point.
                #
                # Add the dimension needed by the model output.
                inputs, labels = (
                    inputs.to(DEVICE),
                    labels.float().to(DEVICE).unsqueeze(1)
                )


                # Run the validation images through the model.
                outputs = model(inputs)


                # Calculate validation loss.
                loss = criterion(
                    outputs,
                    labels
                )


                # Add this batch's loss to the total validation loss.
                val_loss += (
                    loss.item() * inputs.size(0)
                )


                # Convert raw logits into probabilities,
                # then classify them using 0.5 as the threshold.
                #
                # Probability >= 0.5 → 1
                #
                # Probability < 0.5 → 0
                preds = (
                    torch.sigmoid(outputs) >= 0.5
                ).float()


                # Count the number of correct predictions.
                val_correct += (
                    (preds == labels).sum().item()
                )


                # Count the total number of validation examples.
                val_total += labels.size(0)


        # Calculate average validation loss.
        val_loss /= val_total


        # Calculate validation accuracy.
        val_acc = val_correct / val_total


        # Print the training results for the current epoch.
        #
        # epoch+1:
        #     Humans normally count epochs from 1,
        #     while Python range starts at 0.
        #
        # :.4f:
        #     Display the number with 4 decimal places.
        #
        # Example output:
        #
        # Epoch 1/5: Train Loss: 0.3421, Train Acc: 0.8750 |
        # Val Loss: 0.2912, Val Acc: 0.9000
        print(
            f"Epoch {epoch+1}/{EPOCHS}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )


    # =====================================================
    # SAVE TRAINED MODEL
    # =====================================================


    # Create the models directory if it doesn't already exist.
    #
    # exist_ok=True means that no error occurs
    # if the directory already exists.
    os.makedirs(
        'models',
        exist_ok=True
    )


    # Save the trained model's state dictionary.
    #
    # state_dict() contains the model's learned parameters,
    # such as weights and biases.
    #
    # The file will be:
    #
    #     models/mask_detector_pytorch.pth
    torch.save(
        model.state_dict(),
        'models/mask_detector_pytorch.pth'
    )


    # Print a confirmation message.
    print(
        "Model saved to models/mask_detector_pytorch.pth"
    )


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================


# This condition checks whether this file is being
# executed directly.
#
# If you run:
#
#     python train_pytorch.py
#
# then __name__ will be "__main__".
#
# If this file is imported by another Python file,
# this condition will be False.
if __name__ == '__main__':


    # Start the training process by calling main().
    main()
