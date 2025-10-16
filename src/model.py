import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet50_model(num_classes=10):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False # freeze all pretrained layers
    
    for param in model.layer4.parameters(): # unfreeze last layer
        param.requires_grad = True
    
    # replace final layer in CIFAR-10
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model