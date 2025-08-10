import sys
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture and weights
model = mobilenet_v2(pretrained=False)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(model.last_channel, 1)
)
model.load_state_dict(torch.load('models/mask_detector_pytorch.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Preprocessing transforms (must match training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img_path = sys.argv[1]
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Unable to read image {img_path}")
    sys.exit(1)

input_tensor = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(input_tensor)
    prob = torch.sigmoid(output).item()

label = "Mask" if prob < 0.5 else "No Mask"
print(f"Prediction: {label} (probability={prob:.4f})")
