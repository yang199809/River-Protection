# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""River protection segmentation neck modules."""

from typing import List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = (
    "GRN",
    "MSFFM_YOLO",
    "SemanticGuideFusion",
    "SGFPN3",
    "FeatureSelect",
    "ChannelContextAggregationYOLO",
    "RiverDCPBlock",
    "RSE3",
)


def _autopad(k: Union[int, Sequence[int]]) -> Union[int, tuple]:
    """Return padding that preserves spatial size for odd kernels."""
    return k // 2 if isinstance(k, int) else tuple(x // 2 for x in k)


class _ConvBNAct(nn.Module):
    """A compact Conv-BN-Activation block used by SGFPN3."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: Union[int, Sequence[int]] = 1,
        s: int = 1,
        g: int = 1,
        act: Union[bool, nn.Module] = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization and activation."""
        return self.act(self.bn(self.conv(x)))


class GRN(nn.Module):
    """Global response normalization for NCHW feature maps."""

    def __init__(self, in_channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize global channel responses and preserve the input residual."""
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class MSFFM_YOLO(nn.Module):
    """Multi-scale feature fusion module with depthwise context branches."""

    def __init__(self, in_channels: int, out_channels: int = None, kernel_size: int = 9, expand_ratio: int = 2):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-size depthwise convolution.")

        out_channels = out_channels or in_channels
        hidden_channels = int(out_channels * expand_ratio)
        self.expand = _ConvBNAct(in_channels, hidden_channels, 1, act=nn.GELU())
        self.dw3 = _ConvBNAct(hidden_channels, hidden_channels, 3, g=hidden_channels, act=False)
        self.dwk = _ConvBNAct(hidden_channels, hidden_channels, kernel_size, g=hidden_channels, act=False)
        self.dw_h = _ConvBNAct(hidden_channels, hidden_channels, (5, 1), g=hidden_channels, act=False)
        self.dw_w = _ConvBNAct(hidden_channels, hidden_channels, (1, 5), g=hidden_channels, act=False)
        self.act = nn.GELU()
        self.grn = GRN(hidden_channels)
        self.project = _ConvBNAct(hidden_channels, out_channels, 1, act=False)
        self.shortcut = (
            nn.Identity() if in_channels == out_channels else _ConvBNAct(in_channels, out_channels, 1, act=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance local, large-kernel and strip-context features."""
        y = self.expand(x)
        y = y + self.dw3(y) + self.dwk(y) + self.dw_h(y) + self.dw_w(y)
        y = self.grn(self.act(y))
        return self.project(y) + self.shortcut(x)


class SemanticGuideFusion(nn.Module):
    """Fuse low-level features under high-level semantic guidance."""

    def __init__(self, channels: int, kernel_size: int = 9):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-size depthwise convolution.")

        self.att = nn.Sequential(
            _ConvBNAct(channels, channels, 1, act=nn.GELU()),
            _ConvBNAct(channels, channels, kernel_size, g=channels, act=False),
            nn.Sigmoid(),
        )
        self.key = _ConvBNAct(channels, channels, 1, act=False)
        self.value = _ConvBNAct(channels, channels, 1, act=False)
        self.proj = _ConvBNAct(channels, channels, 1, act=False)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """Apply out = proj(att(high) * key(low) + value(low)) + low."""
        if high.shape[2:] != low.shape[2:]:
            high = F.interpolate(high, size=low.shape[2:], mode="bilinear", align_corners=False)
        return self.proj(self.att(high) * self.key(low) + self.value(low)) + low


class SGFPN3(nn.Module):
    """Semantic-guided three-level FPN neck for YOLO11 segmentation heads."""

    def __init__(
        self, in_channels: Sequence[int], out_channels: int = 256, kernel_size: int = 9, expand_ratio: int = 2
    ):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("SGFPN3 expects three input feature levels: [P3, P4, P5].")

        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.lateral_convs = nn.ModuleList(_ConvBNAct(c, out_channels, 1) for c in in_channels)
        self.p5_to_p4 = SemanticGuideFusion(out_channels, kernel_size)
        self.p4_to_p3 = SemanticGuideFusion(out_channels, kernel_size)
        self.down_p3 = _ConvBNAct(out_channels, out_channels, 3, s=2)
        self.down_p4 = _ConvBNAct(out_channels, out_channels, 3, s=2)
        self.msffm = nn.ModuleList(
            MSFFM_YOLO(out_channels, out_channels, kernel_size, expand_ratio) for _ in range(3)
        )

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Resize x to ref spatial size when stride rounding produces a mismatch."""
        return (
            x
            if x.shape[2:] == ref.shape[2:]
            else F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)
        )

    def forward(self, x: Union[List[torch.Tensor], tuple]) -> List[torch.Tensor]:
        """Forward [P3, P4, P5] and return [P3_out, P4_out, P5_out]."""
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("SGFPN3.forward expects a list or tuple with three tensors: [P3, P4, P5].")

        p3, p4, p5 = [conv(feat) for conv, feat in zip(self.lateral_convs, x)]
        p4 = self.p5_to_p4(p4, p5)
        p3 = self.p4_to_p3(p3, p4)

        p3 = self.msffm[0](p3)
        p4 = self.msffm[1](p4 + self._resize_like(self.down_p3(p3), p4))
        p5 = self.msffm[2](p5 + self._resize_like(self.down_p4(p4), p5))
        return [p3, p4, p5]


class ChannelContextAggregationYOLO(nn.Module):
    """Channel context aggregation using pure PyTorch ops for YOLO feature maps."""

    def __init__(self, channels: int, reduction: int = 1, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.inter_channels = max(channels // reduction, 1)
        self.eps = eps
        self.key = nn.Conv2d(channels, 1, 1)
        self.value = nn.Conv2d(channels, self.inter_channels, 1)
        self.proj = nn.Conv2d(self.inter_channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate global context and return a context map with the same shape as x."""
        b, _, h, w = x.shape
        k = self.key(x).view(b, 1, -1, 1).softmax(dim=2)
        v = self.value(x).view(b, 1, self.inter_channels, -1)
        context = torch.matmul(v, k).view(b, self.inter_channels, 1, 1)
        return self.proj(context).expand(-1, -1, h, w)


class RiverDCPBlock(nn.Module):
    """Directional context perception block for long river-structure features."""

    def __init__(self, channels: int, large_size: int = 5, strip_size: int = 9):
        super().__init__()
        if large_size % 2 == 0 or strip_size % 2 == 0:
            raise ValueError("large_size and strip_size must be odd for same-size depthwise convolution.")

        self.channel_context = ChannelContextAggregationYOLO(channels)
        self.large_dw = nn.Conv2d(channels, channels, large_size, padding=large_size // 2, groups=channels, bias=False)
        self.h_strip = nn.Conv2d(
            channels, channels, (1, strip_size), padding=(0, strip_size // 2), groups=channels, bias=False
        )
        self.v_strip = nn.Conv2d(
            channels, channels, (strip_size, 1), padding=(strip_size // 2, 0), groups=channels, bias=False
        )
        self.spatial_bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.gate_conv = nn.Conv2d(channels, channels, 1)
        self.proj = _ConvBNAct(channels, channels, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance strip-like structures with channel context and directional gates."""
        identity = x
        channel_context = self.channel_context(x)
        spatial = self.large_dw(x)
        spatial = spatial + self.h_strip(spatial) + self.v_strip(spatial)
        spatial = self.act(self.spatial_bn(spatial))
        gate = self.gate_conv(spatial).sigmoid()
        return self.proj(channel_context * gate) + identity


class RSE3(nn.Module):
    """River-structure strip enhancement for three-level YOLO segmentation features."""

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int = 256,
        large_size: int = 5,
        strip_size: int = 9,
        mode: str = "p3p4",
    ):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels] * 3
        elif len(in_channels) == 1 and isinstance(in_channels[0], (list, tuple)):
            in_channels = list(in_channels[0])
        else:
            in_channels = list(in_channels)
        if len(in_channels) != 3:
            raise ValueError("RSE3 expects three input feature levels: [P3, P4, P5].")
        if mode not in {"p3", "p3p4", "all"}:
            raise ValueError("mode must be one of {'p3', 'p3p4', 'all'}.")

        enhance_indices = {"p3": {0}, "p3p4": {0, 1}, "all": {0, 1, 2}}[mode]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        self.input_projs = nn.ModuleList(
            nn.Identity() if c == out_channels else _ConvBNAct(c, out_channels, 1) for c in in_channels
        )
        self.blocks = nn.ModuleList(
            RiverDCPBlock(out_channels, large_size, strip_size) if i in enhance_indices else nn.Identity()
            for i in range(3)
        )

    def forward(self, x: Union[List[torch.Tensor], tuple]) -> List[torch.Tensor]:
        """Forward [P3, P4, P5] and return strip-enhanced [P3_out, P4_out, P5_out]."""
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("RSE3.forward expects a list or tuple with three tensors: [P3, P4, P5].")

        return [block(proj(feat)) for block, proj, feat in zip(self.blocks, self.input_projs, x)]


class FeatureSelect(nn.Module):
    """Select a feature tensor from a list returned by SGFPN3."""

    def __init__(self, index: int = 0):
        super().__init__()
        self.index = int(index)

    def forward(self, x: Union[List[torch.Tensor], tuple]) -> torch.Tensor:
        """Return x[index]."""
        return x[self.index]
