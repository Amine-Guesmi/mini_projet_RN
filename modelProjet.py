#Load libraries
import os
import numpy as np
import torch
import glob
from PIL import Image
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib

classes = ['Benign', 'malignant', 'Normal']

#checking for device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=10240, 
                      out_features=output_shape)
        )
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.dropout(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

from pathlib import Path

# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("models")

MODEL_NAME = "model_2_20ep_adam_dropout.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

loaded_model = ModelV2(input_shape=3, 
    hidden_units=10, 
    output_shape=len(classes)).to(device)

# Load state_dict
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

transform=transforms.Compose([
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])

def prediction(image_path):
  img = Image.open(image_path).convert('RGB')
  input = transform(img)
  input = input.unsqueeze(0)
  loaded_model.eval()
  output = loaded_model(input)
  pred_prob = torch.softmax(output.squeeze(), dim=0)
  pred = pred_prob.argmax()
  return classes[pred]