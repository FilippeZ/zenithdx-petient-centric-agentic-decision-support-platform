#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
from torchvision import models

def build_resnet50(num_classes: int,
                   pretrained: bool = True,
                   freeze_backbone: bool = False) -> nn.Module:
    net = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    if freeze_backbone:
        for p in net.parameters(): p.requires_grad = False
    net.fc = nn.Sequential(nn.Dropout(0.3),
                           nn.Linear(net.fc.in_features, num_classes))
    return net
