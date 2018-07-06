import torch 
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, num_features, num_classes):
        super(AttentionLayer, self).__init__()
        self.conv_1 = nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(num_classes)
        # self.bn_2 = nn.BatchNorm2d(num_classes*2)
        # self.conv_2 = nn.Conv2d(num_features, num_classes, kernel_size=3, stride=1, dilation=2, padding=0, bias=True)
        # self.conv_2 = nn.ConvTranspose2d(num_classes*2, num_classes, kernel_size=3, stride=1, padding=0, bias=True)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Sequential(
                # nn.Linear(num_classes, num_classes),
                # nn.ReLU(inplace=True),
                # nn.Linear(num_classes, num_classes),
                nn.Sigmoid()
                # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_1(x)
        # y = self.conv_2(x_input)
        x = self.bn(x)
        # y = self.bn(y)
        # x = self.conv_3(x)
        # x = self.bn_2(x)
        # x = self.conv_2(x)
        # x = self.bn(x)
        b, c, _, _ = x.size()
        # x = self.max_pool(x)
        # y = self.avg_pool(y)
        x = self.max_pool(x).view(b, c)
        # x = self.avg_pool(x).view(b, c)
        # x = x + y
        x = self.attn(x).view(b, c, 1, 1)
        # x = self.fc(x)
        return x 
