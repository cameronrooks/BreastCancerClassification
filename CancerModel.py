import torch
import torch.nn as nn
import torch.nn.functional as F

class CancerModel(nn.Module):


    def __init__(self):
        super(CancerModel, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = torch.flatten(x)
        x = self.fc(x)
  
        return x
