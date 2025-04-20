import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np

# ---------------------------
# Define the Emotion Classifier
# ---------------------------
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(EmotionClassifier, self).__init__()
        self.base_model = models.resnet18(pretrained=False)
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    model = EmotionClassifier(num_classes=3)
    model.load_state_dict(torch.load("emotion_classifier.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ---------------------------
# Define transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ---------------------------
# Label Mapping
# ---------------------------
# Adjust these to your actual labels
labels_map = {
    0: "Happy",
    1: "Sad",
    2: "Angry"
}

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Emotion Classifier from Image")
st.write("Upload an image of a face and I’ll tell you the emotion!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
        st.success(f"Predicted Emotion: **{labels_map[predicted_class]}**")
