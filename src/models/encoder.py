import torch
import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    """Encoder that outputs spatial image features for attention.

    Uses ResNet-101 pretrained; returns features shaped (batch, num_pixels, embed_dim)
    and feature_map shaped (batch, C, H, W) for visualization.
    """

    def __init__(self, embed_dim=512, fine_tune=False):
        super().__init__()
        resnet = models.resnet101(pretrained=True)
        # remove avgpool and fc
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.conv_proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
        self.fine_tune(fine_tune)

    def forward(self, images):
        # images: (B,3,H,W)
        feat_map = self.backbone(images)  # (B,2048,Hf,Wf)
        feat_map = self.adaptive_pool(feat_map)  # (B,2048,14,14)
        proj = self.conv_proj(feat_map)  # (B,embed,14,14)
        B, C, H, W = proj.size()
        features = proj.view(B, C, H * W).permute(0, 2, 1)  # (B, num_pixels, embed)
        return features, feat_map

    def fine_tune(self, fine_tune=False):
        for p in self.backbone.parameters():
            p.requires_grad = fine_tune
