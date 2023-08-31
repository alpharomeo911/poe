#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class UNetModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.copy1 = None
        self.copy2 = None
        self.copy3 = None
        self.copy4 = None
        # Encoding
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3))
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3))
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3))
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3))
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3))
        self.conv10 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3))
        # Decoding
        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3,3))
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3))
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3,3))
        self.conv14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3))
        self.conv15 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3))
        self.conv16 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3))
        self.conv17 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3))
        self.conv18 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3))
        self.conv19 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1,1))



    def forward(self, x):
        # Encoding
        x = F.max_pool2d(F.relu(self.conv2(F.relu(self.conv1(x)))), kernel_size=(2,2), stride=(2,2))
        self.copy1 = x.detach().clone()
        x = F.max_pool2d(F.relu(self.conv4(F.relu(self.conv3(x)))), kernel_size=(2,2), stride=(2,2))
        self.copy2 = x.detach().clone()
        x = F.max_pool2d(F.relu(self.conv6(F.relu(self.conv5(x)))), kernel_size=(2,2), stride=(2,2))
        self.copy3 = x.detach().clone()
        x = F.max_pool2d(F.relu(self.conv8(F.relu(self.conv7(x)))), kernel_size=(2,2), stride=(2,2))
        self.copy4 = x.detach().clone()
        x = F.max_pool2d(F.relu(self.conv10(F.relu(self.conv9(x)))), kernel_size=(2,2), stride=(2,2))
        # Decoding
        x = F.relu(self.conv12(F.relu(self.conv11(F.conv_transpose2d(x, kernel_size=(2,2), stride=(2,2))))))+self.copy4
        x = F.relu(self.conv14(F.relu(self.conv13(F.conv_transpose2d(x, kernel_size=(2,2), stride=(2,2))))))+self.copy3
        x = F.relu(self.conv16(F.relu(self.conv15(F.conv_transpose2d(x, kernel_size=(2,2), stride=(2,2))))))+self.copy2
        x = F.relu(self.conv18(F.relu(self.conv17(F.conv_transpose2d(x, kernel_size=(2,2), stride=(2,2))))))+self.copy1
        x = self.conv19(x)
        