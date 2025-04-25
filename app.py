pip install gradio
import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image

# Configuration
IMG_SIZE = 224
NUM_CLASSES = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing for input images
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# CNN model architecture
class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load model
model = CustomCNN().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Emotion labels
emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Prediction function
def predict_emotion(image):
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        output_values = outputs[0].cpu().numpy()
        top_5_indices = np.argsort(output_values)[-5:][::-1]
        top_5_emotions = [emotion_labels[i] for i in top_5_indices]
        top_5_values = [output_values[i] for i in top_5_indices]
    
    result = "\n".join([f"{emotion}: {value:.2f}" for emotion, value in zip(top_5_emotions, top_5_values)])
    return f"Top 5 Emotion Scores:\n{result}"

# Gradio Interface
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Emotion Prediction"),
    title="Emotion Prediction from Image",
    description="Upload an image to predict the top 5 emotions with the highest confidence scores."
)

# Launch
if __name__ == "__main__":
    interface.launch(share=True)
