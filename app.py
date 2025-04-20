import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import kagglehub

# Define your model architecture
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(EmotionClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Streamlit UI
st.title("😊 Emotion Classifier")
st.write("Upload a face image to predict the emotion.")

# Download the latest version of the model from Kaggle Hub
path = kagglehub.model_download("prakhar146/emotion-classification/pyTorch/default")
st.write(f"Model loaded from: {path}")

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionClassifier(num_classes=3).to(device)
model.load_state_dict(torch.load(path, map_location=device))
model.eval()

# Define transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Class labels (adjust according to your dataset)
label_map = {0: "Angry", 1: "Sad", 2: "Happy"}

# Image uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image and make a prediction
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        emotion = label_map[predicted.item()]
    
    st.success(f"Predicted Emotion: **{emotion}**")
