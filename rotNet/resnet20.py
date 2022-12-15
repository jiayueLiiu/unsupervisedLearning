import torch
import torch.nn as nn
import torch.nn.functional as F

# from FP_layers import *

class ResNet_Block(nn.Module):
    def __init__(self, in_chs, out_chs, strides, Nbits=None):
        super(ResNet_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chs, out_channels=out_chs,
                      stride=strides, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_chs, out_channels=out_chs,
                      stride=1, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_chs))

        if in_chs != out_chs:
            self.id_mapping = nn.Sequential(
                nn.Conv2d(in_channels=in_chs, out_channels=out_chs,
                          stride=strides, padding=0, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_chs))
        else:
            self.id_mapping = None
        self.final_activation = nn.ReLU(True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.id_mapping is not None:
            x_ = self.id_mapping(x)
        else:
            x_ = x
        return self.final_activation(x_ + out)

class ResNetCIFAR(nn.Module):
    def __init__(self, num_layers=20, Nbits=None):
        super(ResNetCIFAR, self).__init__()
        self.num_layers = num_layers
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      stride=1, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        num_layers_per_stage = (num_layers - 2) // 6
        self.body_op = []
        num_inputs = 16
        # Stage 1
        for j in range(num_layers_per_stage):
            strides = 1
            self.body_op.append(ResNet_Block(num_inputs, 16, strides))
            num_inputs = 16
        # Stage 2
        for j in range(num_layers_per_stage):
            if j == 0:
                strides = 2
            else:
                strides = 1
            self.body_op.append(ResNet_Block(num_inputs, 32, strides))
            num_inputs = 32
        # Stage 2
        for j in range(num_layers_per_stage):
            if j == 0:
                strides = 2
            else:
                strides = 1
            self.body_op.append(ResNet_Block(num_inputs, 64, strides))
            num_inputs = 64
            
        self.body_op = nn.Sequential(*self.body_op)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.final_fc_unsup = nn.Linear(64, 4)
        self.final_fc_sup = nn.Linear(64, 10)

    def forward(self, x):
        out = self.head_conv(x)
        out = self.body_op(out)
        self.features = self.avg_pool(out)
        self.feat_1d = self.features.mean(3).mean(2)
        return self.final_fc_unsup(self.feat_1d), self.final_fc_sup(self.feat_1d)

