import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2

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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = transform(frame).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    label = "Mask" if prob < 0.5 else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    cv2.putText(frame, f"{label} ({prob:.2f})", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.imshow("Face Mask Detector (PyTorch)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
