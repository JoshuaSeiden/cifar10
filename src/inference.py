import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import argparse
import os

# CIFAR 10 classes
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_model(weights_path, num_classes=10, device='cpu'):
    # Load fine-tuned ResNet18 model for inference
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    #model.fc = nn.Linear(512, 10)

    # Load trained weights
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model


def preprocess_image(image_path):
    # Standard CIFAR-10 preprocessing
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0) # add batch dim

def predict(model, image_tensor, device='cpu'):
    # Run inference and return top class
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
    return preds.item()

def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Inference Script")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--image", type=str, required=True, help="Path to an image or folder of images")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, num_classes=10, device=device)

    if os.path.isdir(args.image):
        image_files = [f for f in os.listdir(args.image) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_name in image_files:
            img_path = os.path.join(args.image, img_name)
            img_tensor = preprocess_image(img_path)
            pred_idx = predict(model, img_tensor, device)
            print(f"Prediction: {CLASS_NAMES[pred_idx]}")
    else:
        img_tensor = preprocess_image(args.image)
        pred_idx = predict(model, img_tensor, device)
        print(f"Prediction: {CLASS_NAMES[pred_idx]}")

if __name__ == "__main__":
    main()

# example usage: python inference.py --weights models/cifar10_resnet18.pth --image data/test/cat1.jpg