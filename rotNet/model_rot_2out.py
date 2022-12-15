import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.fc_sup = nn.Linear(512,10)
        self.fc_unsup = nn.Linear(512,4)
        #self.g = nn.Sequential(nn.Linear(512, 128, bias=False), nn.BatchNorm1d(128),
        #                       nn.ReLU(inplace=True), nn.Linear(128, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        #out = feature#self.g(feature)
        # return F.normalize(feature, dim=-1)#, F.normalize(out, dim=-1)
        return self.fc_unsup(feature), self.fc_sup(feature)
