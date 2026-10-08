"""CPU inference using the architecture saved by the Kaggle ResNet notebook."""
import numpy as np
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, stride=stride, padding=2, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            ) if in_channels != out_channels or stride != 1 else nn.Identity()
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.main(x) + self.shortcut(x))


class ECGResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(12, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            ResidualBlock(32, 32), ResidualBlock(32, 64, 2),
            ResidualBlock(64, 128, 2), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 5))

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model(path, classes):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != list(classes):
        raise ValueError("模型类别顺序与演示数据不一致")
    model = ECGResNet()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def predict(model, standardized_waveforms):
    waveforms = np.array(standardized_waveforms, dtype=np.float32, copy=True)
    if waveforms.ndim == 2:
        waveforms = waveforms[None]
    if waveforms.ndim != 3 or waveforms.shape[1:] != (12, 1000):
        raise ValueError("输入必须为 (样本数, 12, 1000)")
    if not np.isfinite(waveforms).all():
        raise ValueError("波形包含无效值")
    with torch.inference_mode():
        return torch.sigmoid(model(torch.from_numpy(waveforms))).numpy()
