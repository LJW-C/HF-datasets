from __future__ import annotations

import math

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IQCNN(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 2, width: int = 64, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(input_channels, width, 7, 2, dropout),
            ConvBlock(width, width * 2, 7, 2, dropout),
            ConvBlock(width * 2, width * 2, 5, 2, dropout),
            ConvBlock(width * 2, width * 4, 5, 2, dropout),
            ConvBlock(width * 4, width * 4, 3, 2, dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = dilation
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch))
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 2, width: int = 64, dropout: float = 0.1):
        super().__init__()
        self.stem = ConvBlock(input_channels, width, 9, 2, dropout)
        self.body = nn.Sequential(
            ResidualBlock(width, width, 1, 1, dropout),
            ResidualBlock(width, width * 2, 2, 1, dropout),
            ResidualBlock(width * 2, width * 2, 1, 2, dropout),
            ResidualBlock(width * 2, width * 4, 2, 1, dropout),
            ResidualBlock(width * 4, width * 4, 1, 2, dropout),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width * 4, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(x)))


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCN(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 2, width: int = 96, dropout: float = 0.15):
        super().__init__()
        self.input = ConvBlock(input_channels, width, 7, 2, dropout)
        self.blocks = nn.Sequential(*[TCNBlock(width, 2**i, dropout) for i in range(7)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.input(x)))


class CLDNN(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 2, width: int = 64, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(input_channels, width, 7, 2, dropout),
            ConvBlock(width, width * 2, 5, 2, dropout),
            ConvBlock(width * 2, width * 2, 5, 2, dropout),
        )
        self.gru = nn.GRU(width * 2, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Dropout(dropout), nn.Linear(hidden * 2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x).transpose(1, 2)
        z, _ = self.gru(z)
        return self.head(z.mean(dim=1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerIQ(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 2,
        d_model: int = 128,
        heads: int = 4,
        layers: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv1d(input_channels, d_model, kernel_size=16, stride=8, padding=4, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.pos = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.patch(x).transpose(1, 2)
        z = self.encoder(self.pos(z))
        return self.head(z.mean(dim=1))


class ConformerBlock1D(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Sequential(
            nn.Conv1d(d_model, d_model, 15, padding=7, groups=d_model, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.attn_norm(x)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        x = x + attn_out
        z = self.conv_norm(x).transpose(1, 2)
        x = x + self.depthwise(z).transpose(1, 2)
        return x + self.ffn(x)


class ConformerIQ(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 2, d_model: int = 128, heads: int = 4, layers: int = 4, dropout: float = 0.15):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv1d(input_channels, d_model, kernel_size=16, stride=8, padding=4, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(d_model)
        self.blocks = nn.Sequential(*[ConformerBlock1D(d_model, heads, dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.patch(x).transpose(1, 2)
        z = self.blocks(self.pos(z))
        return self.head(z.mean(dim=1))


def build_model(
    name: str,
    num_classes: int,
    input_channels: int = 2,
    width: int = 64,
    dropout: float = 0.15,
) -> nn.Module:
    name = name.lower()
    if name == "iq_cnn":
        return IQCNN(num_classes, input_channels, width=width, dropout=dropout)
    if name == "resnet1d":
        return ResNet1D(num_classes, input_channels, width=width, dropout=dropout)
    if name == "tcn":
        return TCN(num_classes, input_channels, width=max(width, 64), dropout=dropout)
    if name == "cldnn":
        return CLDNN(num_classes, input_channels, width=width, dropout=dropout)
    if name == "transformer":
        return TransformerIQ(num_classes, input_channels, d_model=max(width * 2, 96), dropout=dropout)
    if name == "conformer":
        return ConformerIQ(num_classes, input_channels, d_model=max(width * 2, 96), dropout=dropout)
    raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = ["iq_cnn", "resnet1d", "tcn", "cldnn", "transformer", "conformer"]

