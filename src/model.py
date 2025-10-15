import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet18_model(num_classes=10):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False # freeze all pretrained layers
    
    # replace final layer in CIFAR-10
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model