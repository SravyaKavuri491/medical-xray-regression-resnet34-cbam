"""
Model Module for Medical Image Regression Task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ============================================================
# SIMPLE CNN BASELINE
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze(1)

# ============================================================
# CBAM ATTENTION (Correct Version)
# ============================================================

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )

        # Spatial Attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        avg = self.avg_pool(x).view(b, c)
        maxv = self.max_pool(x).view(b, c)

        ca = self.sigmoid(self.mlp(avg) + self.mlp(maxv)).view(b, c, 1, 1)
        x = x * ca

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        sa = self.spatial(torch.cat([avg_out, max_out], dim=1))

        return x * sa

# ============================================================
# STUDENT MODEL: ResNet34 + CBAM + Feature Fusion
# ============================================================

class StudentModel(nn.Module):
    def __init__(self, num_channels=3, pretrained=True):
        super(StudentModel, self).__init__()

        base = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.layer1 = base.layer1  # 64
        self.layer2 = base.layer2  # 128
        self.layer3 = base.layer3  # 256
        self.layer4 = base.layer4  # 512

        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        self.reduce3 = nn.Conv2d(256, 128, 1)
        self.reduce4 = nn.Conv2d(512, 128, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.regressor = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x3 = self.layer3(x)
        x4 = self.layer4(x3)

        x3 = self.cbam3(x3)
        x4 = self.cbam4(x4)

        f3 = self.pool(self.reduce3(x3)).view(x.size(0), -1)
        f4 = self.pool(self.reduce4(x4)).view(x.size(0), -1)

        feat = torch.cat([f3, f4], dim=1)

        return self.regressor(feat).squeeze(1)

# ============================================================
# FACTORY
# ============================================================

def get_model(model_name='simple_cnn', **kwargs):
    if model_name == 'simple_cnn':
        return SimpleCNN(**kwargs)
    elif model_name == 'student':
        return StudentModel(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ============================================================

if __name__ == '__main__':
    model = StudentModel()
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print("Output shape:", out.shape)
    print("Params:", sum(p.numel() for p in model.parameters()))
