# model.py
import torch
import torch.nn as nn
from torchvision import models

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel_size=7,padding=3,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        max_pool = torch.max(x,dim=1,keepdim=True)[0]
        avg_pool = torch.mean(x,dim=1,keepdim=True)
        pooled = torch.cat([max_pool, avg_pool], dim=1)
        attn = self.sigmoid(self.conv(pooled))
        return x * attn

class SSCEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    def forward(self,x):
        x = self.ca(x)
        x = self.sa(x)
        return x

def build_efficientnet_b0(num_classes, dropout=0.3):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # change first conv to accept 2 channels
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(2, first_conv.out_channels,
                                     kernel_size=first_conv.kernel_size,
                                     stride=first_conv.stride,
                                     padding=first_conv.padding,
                                     bias=False)
    # insert SSCE after first conv (out channels = first_conv.out_channels)
    in_ch = model.features[0][0].out_channels
    model.features.insert(1, SSCEBlock(in_ch))
    # replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    return model
