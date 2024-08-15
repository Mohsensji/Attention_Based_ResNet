import torch
import torch.nn as nn
import torch.nn.functional as F

class _Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(_Block, self).__init__()
        self.layers = []
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.layers.append(nn.BatchNorm2d(out_channels))

        self.res = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
        
        self.sequential_layers = nn.Sequential(*self.layers)
    def forward(self, x):
        out = self.sequential_layers(x)
        out += self.res(x)
        out = F.relu(out)
        return out

class AttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        return x * out.expand_as(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.layers = []
        
        self.layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False))
        self.layers.append(nn.BatchNorm2d(in_channels))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False))
        self.layers.append(nn.BatchNorm2d(in_channels))
        self.layers.append(AttentionModule(in_channels))
        self.sequential_layers = nn.Sequential(*self.layers)
    def forward(self, x):
        residual = x
        out = self.sequential_layers(x)
        out += residual
        out = F.relu(out)
        return out

class MyCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(MyCNN, self).__init__()
        self.layers = []
        self.in_channels = 64

        self.layers.append(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(nn.ReLU())
        self.layers.append(self._make_layer(_Block, 64, 2, stride=1))
        self.layers.append(self._make_layer(_Block, 128, 2, stride=2))
        self.layers.append(self._make_layer(_Block, 256, 2, stride=2))
        self.layers.append(AttentionBlock(256))
        self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(256, num_classes)
        self.sequential_layers = nn.Sequential(*self.layers)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.sequential_layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = MyCNN(num_classes=20)