import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# CNN class must match training exactly
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

# Instantiate model and load weights
model = CNN()
model.load_state_dict(torch.load("cnn_deepfake.pth", map_location=torch.device("cpu")))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # match training size
    transforms.ToTensor(),
])

# Load image
image_path = "C:/Users/sinzs/Downloads/test.png"  # your image path
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Predict
with torch.no_grad():
    output = model(image_tensor)
    _, prediction = torch.max(output, 1)
    label = "Fake" if prediction.item() == 0 else "Real"
    print(f"🧠 Prediction: {label}")
