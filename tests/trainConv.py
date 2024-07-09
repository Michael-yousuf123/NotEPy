import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

transform_data = transforms.Compose([transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class ConNet(nn.Module):
    """
    """
    def __init__(self):
        super(ConNet, self).__init__()

        # layer 1
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=5, stride = 1, padding=2)
        self.batch1 = nn.BatchNorm2d()
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # layer 2
        self.conv2 = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=5, stride = 1, padding=2)
        self.batch2 = nn.BatchNorm2d()
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # full connected layer
        self.fcl = nn.Linear()
    def forward(self, x):
        """
        """
        # Conv 1
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu1(out)
        
        # Max Pool 1
        out = self.pool1(out)
        
        # Conv 2
        out = self.conv2(x)
        out = self.batch2(out)
        out = self.relu2(out)

        # Max Pool 2
        out = self.pool2(out)
        
        out = out.view(out.size(0), -1)
        # Linear Layer
        out = self.fcl(out)
        return out