"""
models/cnn.py
-------------
Convolutional architectures for WiFi CSI activity classification.

Both models treat a CSI amplitude (or complex) matrix as a single-channel
2-D image where the spatial axes are *time* (rows) and *subcarrier index*
(columns).  This lets standard 2-D convolutions capture:

- **Temporal patterns** along the time axis (motion dynamics).
- **Frequency correlations** along the subcarrier axis (multipath structure).

Typical input tensor shape: ``(B, 1, T, S)``
  - B  = batch size
  - 1  = channel (amplitude magnitude; extend to 2 for complex I/Q)
  - T  = time steps, e.g. 250 (≈ 1 s at 250 Hz packet rate)
  - S  = number of subcarriers, e.g. 90 (30 sub-carriers × 3 antennae)
"""

import torch
import torch.nn as nn
from typing import List


# ---------------------------------------------------------------------------
# Utility blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU building block.

    Using BatchNorm before ReLU stabilises training with the relatively small
    batch sizes common in CSI experiments (CSI datasets are often a few
    thousand samples).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,  # BatchNorm absorbs the bias term
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


# ---------------------------------------------------------------------------
# CSICNN
# ---------------------------------------------------------------------------

class CSICNN(nn.Module):
    """Lightweight 3-block CNN for CSI activity classification.

    Architecture
    ------------
    Three convolutional blocks, each consisting of:
      - Conv2d (3×3, same padding) → BatchNorm → ReLU
      - MaxPool2d (2×2, stride 2)

    The growing channel depth (32 → 64 → 128) follows standard practice:
    early layers capture fine-grained local patterns (individual subcarrier
    fluctuations); deeper layers aggregate richer, more abstract features.

    A 2-D adaptive average pool collapses arbitrary T×S feature maps to 1×1,
    making the model input-size agnostic.  A single fully-connected layer with
    dropout produces the final class logits.

    Parameters
    ----------
    num_classes : int
        Number of output activity classes (e.g. 5 for empty/walk/sit/stand/fall).
    in_channels : int, optional
        Number of input channels.  Use 1 for amplitude-only input, 2 for
        complex (I+Q stacked), etc.  Default: 1.
    base_channels : int, optional
        Channel width of the first conv block; doubled in each subsequent
        block.  Default: 32.
    dropout : float, optional
        Dropout probability applied before the classification head.
        Default: 0.5.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.block1 = nn.Sequential(
            ConvBnRelu(in_channels, c1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            ConvBnRelu(c1, c2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            ConvBnRelu(c2, c3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Collapse feature maps to (B, c3, 1, 1) regardless of input size.
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c3, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, C, T, S)``.  For default settings: ``(B, 1, 250, 90)``.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(B, num_classes)``.
        """
        x = self.block1(x)   # (B, c1, T/2,  S/2)
        x = self.block2(x)   # (B, c2, T/4,  S/4)
        x = self.block3(x)   # (B, c3, T/8,  S/8)
        x = self.global_pool(x)   # (B, c3, 1, 1)
        return self.classifier(x)  # (B, num_classes)


# ---------------------------------------------------------------------------
# Residual block used by CSIResNet
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Basic residual block with optional projection shortcut.

    Two 3×3 convolutions with the same number of output channels, plus a
    skip connection.  When ``stride > 1`` or ``in_channels != out_channels``
    the shortcut uses a 1×1 conv to match dimensions.

    For CSI data the residual formulation is beneficial because:
    - Deep networks with residuals are easier to train than plain nets.
    - Identity shortcuts allow gradient flow even when the feature map
      changes are subtle (which is common with clean CSI amplitude signals).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = ConvBnRelu(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        # Second conv does not use an activation here; ReLU applied after add.
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        # Projection shortcut when spatial size or channel count changes.
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + identity)
        return out


# ---------------------------------------------------------------------------
# CSIResNet
# ---------------------------------------------------------------------------

class CSIResNet(nn.Module):
    """ResNet-style model for CSI activity classification.

    Architecture
    ------------
    Follows the ResNet-18 blueprint scaled for CSI inputs:

    1. **Stem** – single Conv2d (7×7, stride 2) + BN + ReLU + MaxPool.
       Reduces the initial 250×90 map to ~62×22 before the residual stages.

    2. **Four residual stages**, each containing ``blocks_per_stage``
       (default 2) ``ResidualBlock`` units.  Channel widths double at each
       stage transition (32→64→128→256).  The first block of stages 2–4
       uses stride 2 to downsample.

    3. **Global average pool** → Dropout → Linear classifier.

    Architecture rationale for CSI
    --------------------------------
    - The 7×7 stem is wide enough to capture multi-cycle Doppler waveforms
      spanning ~7 time steps while simultaneously averaging over 7 adjacent
      subcarriers.
    - Four stages of progressively abstracted features mirror the hierarchy
      in optical images: raw fluctuations → local motion → whole-body
      activity patterns.

    Parameters
    ----------
    num_classes : int
        Number of output activity classes.
    in_channels : int, optional
        Input channel count.  Default: 1.
    base_channels : int, optional
        Channel width after the stem; doubles in each stage.  Default: 32.
    blocks_per_stage : int, optional
        Number of ``ResidualBlock`` units per stage.  Default: 2.
    dropout : float, optional
        Dropout rate before the FC head.  Default: 0.5.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        base_channels: int = 32,
        blocks_per_stage: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # ---- Stem ----
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ---- Residual stages ----
        channels: List[int] = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]

        self.stage1 = self._make_stage(channels[0], channels[0], blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(channels[0], channels[1], blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(channels[1], channels[2], blocks_per_stage, stride=2)
        self.stage4 = self._make_stage(channels[2], channels[3], blocks_per_stage, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(channels[3], num_classes),
        )

        self._init_weights()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Create a sequential stage of residual blocks."""
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, C, T, S)``.

        Returns
        -------
        torch.Tensor
            Class logits ``(B, num_classes)``.
        """
        x = self.stem(x)      # (B, base_channels, T/4, S/4)
        x = self.stage1(x)    # (B, C0, T/4,  S/4)
        x = self.stage2(x)    # (B, C1, T/8,  S/8)
        x = self.stage3(x)    # (B, C2, T/16, S/16)
        x = self.stage4(x)    # (B, C3, T/32, S/32)
        x = self.global_pool(x)   # (B, C3, 1, 1)
        return self.classifier(x)  # (B, num_classes)
