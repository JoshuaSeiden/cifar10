import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

# CIFAR 10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Model Loading
@st.cache_resource
def load_model(weights_path, device):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model

# Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = image.convert('RGB')
    return transform(image).unsqueeze(0)

# Prediction
def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
    return preds.item(), conf.item(), probs.cpu().numpy().flatten()

# Streamlit UI
st.set_page_config(page_title="CIFAR-10 Classifier", layout="centered")
st.title("Fine-tuned ResNet50 trained on CIFAR-10")
st.markdown("Upload an image to classify it into one of the **CIFAR-10** categories.")

weights_path = os.path.join("models", "cifar10_resnet50.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = load_model(weights_path, device)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    with st.spinner("Analyzing image..."):
        image_tensor = preprocess_image(image)
        pred_idx, confidence, probs = predict(model, image_tensor, device)
        predicted_class = CLASS_NAMES[pred_idx]

    st.success(f"Prediction: **{predicted_class.capitalize()}** ({confidence*100:.2f}% confidence)")

    # Probability table
    st.subheader("Class Probabilities")
    prob_dict = {CLASS_NAMES[i]: f"{probs[i]*100:.2f}%" for i in range(len(CLASS_NAMES))}
    st.table(prob_dict)
else:
    st.info("Upload an image to get started.")
