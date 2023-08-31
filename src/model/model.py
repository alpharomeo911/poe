#!/usr/bin/python3
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class UNetModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels)


    def forward(self, x):
        pass