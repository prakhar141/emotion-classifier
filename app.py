import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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

# Load model
model = EmotionClassifier(num_classes=3)
model.load_state_dict(torch.load("emotion_classifier.pth", map_location=torch.device('cpu')))
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Class labels (adjust according to your dataset)
label_map = {0: "Anger", 1: "Fear", 2: "Happy"}

# Streamlit UI
st.title("😊 Emotion Classifier")
st.write("Upload a face image to predict the emotion.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        emotion = label_map[predicted.item()]
    
    st.success(f"Predicted Emotion: **{emotion}**")
