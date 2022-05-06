import torch
from __utils__ import *
import torch.nn as nn

# TODO: Copy-paste your RoadDetectorNet from the colab notebook here.
class RoadDetectorNet(nn.Module):
    def __init__(self):
        super(RoadDetectorNet, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3),
        )
        self.fc = nn.Sequential(nn.Linear(16 * (H - 10) * (W - 10), 16), nn.ReLU(inplace=True),
                                nn.Linear(16, 2))

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
