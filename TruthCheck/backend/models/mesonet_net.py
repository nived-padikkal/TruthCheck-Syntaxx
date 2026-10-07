"""MesoNet: a compact facial video forgery detection network.

Reference: Afchar et al., "MesoNet: a Compact Facial Video
Forgery Detection Network", WIFS 2018.

This module provides PyTorch implementations of Meso-4 and
MesoInception-4, plus a small wrapper that produces a fake
probability in [0, 1] which can be ensembled with the
EfficientNet-based detector in inference_2.py.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn

__all__ = ["Meso4", "MesoInception4", "MesoNetDetector"]


def _conv_block(in_c, out_c, kernel_size, pooling=None, dropout=None):
    layers = [nn.Conv2d(in_c, out_c, kernel_size, padding=kernel_size // 2),
              nn.BatchNorm2d(out_c),
              nn.LeakyReLU(0.1, inplace=True)]
    if pooling is not None:
        layers.append(nn.MaxPool2d(pooling))
    if dropout is not None:
        layers.append(nn.Dropout2d(dropout))
    return layers


class Meso4(nn.Module):
    """Meso-4 network. Input: (N, 3, 256, 256) in [0, 1]."""

    def __init__(self):
        super().__init__()
        layers = []
        layers += _conv_block(3, 8, 3)
        layers += _conv_block(8, 8, 3, pooling=2)
        layers += _conv_block(8, 8, 5, pooling=2, dropout=0.5)
        layers += _conv_block(8, 16, 5, pooling=2, dropout=0.5)
        layers += _conv_block(16, 16, 5, pooling=4, dropout=0.5)
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(16 * 8 * 8, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.classifier(x))


class _InceptionBlock(nn.Module):
    def __init__(self, in_c, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_c, out_1x1, 1, padding=0),
                                     nn.BatchNorm2d(out_1x1), nn.LeakyReLU(0.1, inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(in_c, red_3x3, 1, padding=0),
                                     nn.BatchNorm2d(red_3x3), nn.LeakyReLU(0.1, inplace=True),
                                     nn.Conv2d(red_3x3, out_3x3, 3, padding=1),
                                     nn.BatchNorm2d(out_3x3), nn.LeakyReLU(0.1, inplace=True))
        self.branch3 = nn.Sequential(nn.Conv2d(in_c, red_5x5, 1, padding=0),
                                     nn.BatchNorm2d(red_5x5), nn.LeakyReLU(0.1, inplace=True),
                                     nn.Conv2d(red_5x5, out_5x5, 5, padding=2),
                                     nn.BatchNorm2d(out_5x5), nn.LeakyReLU(0.1, inplace=True))
        self.branch4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                     nn.Conv2d(in_c, pool_proj, 1, padding=0),
                                     nn.BatchNorm2d(pool_proj), nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class MesoInception4(nn.Module):
    """MesoInception-4 network. Input: (N, 3, 256, 256) in [0, 1]."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            _InceptionBlock(8, 1, 4, 4, 4, 4, 2),      # out: 1+4+4+2 = 11 ch
            nn.MaxPool2d(2),
            _InceptionBlock(11, 2, 4, 4, 4, 4, 2),     # out: 2+4+4+2 = 12 ch
            nn.MaxPool2d(2),
            _InceptionBlock(12, 4, 2, 4, 2, 4, 2),     # out: 4+4+4+2 = 14 ch
            nn.MaxPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(14 * 8 * 8, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.classifier(x))


class MesoNetDetector:
    """Wraps a MesoNet model for inference.

    Loads weights from ``weights_path`` if the file exists, otherwise the
    model stays untrained and ``available`` is False (callers should then
    rely on the EfficientNet branch only).
    """

    def __init__(self, weights_path="checkpoints/mesonet.pth", architecture="meso4", device="cpu"):
        self.device = torch.device(device)
        if architecture.lower() == "mesoinception4":
            self.model = MesoInception4()
        else:
            self.model = Meso4()
        self.model.to(self.device)
        self.model.eval()
        self.available = False
        if os.path.exists(weights_path):
            try:
                ckpt = torch.load(weights_path, map_location=self.device)
                state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                self.model.load_state_dict(state, strict=False)
                self.available = True
                print(f"[MesoNet] Loaded weights from {weights_path}")
            except Exception as e:
                print(f"[MesoNet] Could not load weights: {e}")
        else:
            print(f"[MesoNet] No weights found at '{weights_path}'. "
                  "MesoNet branch will be skipped (EfficientNet used alone).")

    def _preprocess(self, face_rgb):
        face = cv2.resize(face_rgb, (256, 256)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(face.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def predict_fake_proba(self, face_rgb):
        """Return P(fake) in [0, 1] for a single RGB face crop."""
        if not self.available:
            return None
        with torch.no_grad():
            out = self.model(self._preprocess(face_rgb))
        return float(out.view(-1).cpu().numpy()[0])
