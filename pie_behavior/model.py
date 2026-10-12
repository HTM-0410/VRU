"""CNN + geometry MLP + GRU with independent binary current-behavior heads."""

import torch
from torch import nn

from . import HEADS


class BehaviorModel(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, freeze_backbone=True,
                 hidden_size=128, dropout=0.2):
        super().__init__()
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone and backbone == "resnet18"
        visual_dim = 0
        if backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18
            cnn = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            visual_dim = cnn.fc.in_features
            cnn.fc = nn.Identity()
            self.cnn = cnn
        elif backbone == "tiny":
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            visual_dim = 64
        elif backbone != "none":
            raise ValueError(f"Unknown backbone: {backbone}")
        if self.freeze_backbone:
            self.cnn.requires_grad_(False)
        self.geometry_encoder = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.gru = nn.GRU(visual_dim + 32, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, len(HEADS)))

    def train(self, mode=True):
        super().train(mode)
        # Frozen backbone BatchNorm must not accumulate validation/train drift.
        if self.freeze_backbone:
            self.cnn.eval()
        return self

    def forward(self, geometry, images=None):
        features = self.geometry_encoder(geometry)
        if self.backbone_name != "none":
            if images is None or images.shape[:2] != geometry.shape[:2]:
                raise ValueError("Expected images [B,T,3,H,W] aligned with geometry [B,T,9]")
            b, t = images.shape[:2]
            with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_backbone):
                visual = self.cnn(images.reshape(b * t, *images.shape[2:])).reshape(b, t, -1)
            features = torch.cat((visual, features), dim=-1)
        _, state = self.gru(features)
        return self.head(state[-1])
