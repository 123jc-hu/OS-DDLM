from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from utils.config import load_config


def _scale_kernel(k_base: int, fs_new: int, fs_ref: int = 256, min_k: int = 3) -> int:
    """按采样率把核长从参考fs缩放到当前fs，并保证为奇数且>=min_k"""
    k = max(min_k, int(round(k_base * fs_new / fs_ref)))
    return k if k % 2 == 1 else k + 1


class TemporalConvBlock(nn.Module):
    """
    三分支“每通道时间卷积”：
      输入:  [B, 1, C, T]
      输出:  [B, F_temporal, C, T]  (默认 F_temporal=24)
    """
    def __init__(self, n_input_chan: int, fs: int,
                 branch_out: int = 8,               # 每分支每通道的输出通道数
                 k_bases=(16, 32, 64), fs_ref: int = 256):
        super().__init__()
        self.n_input_chan = n_input_chan
        self.branch_out = branch_out

        ks = [_scale_kernel(k, fs, fs_ref) for k in k_bases]  # 自适应核长
        convs, bns = [], []
        for k in ks:
            convs.append(
                nn.Conv2d(
                    in_channels=n_input_chan, out_channels=n_input_chan * branch_out,
                    kernel_size=(1, k), padding=(0, k // 2), groups=n_input_chan, bias=False
                )
            )
            bns.append(nn.BatchNorm2d(n_input_chan * branch_out))
        self.convs = nn.ModuleList(convs)
        self.bns   = nn.ModuleList(bns)

        # concat 后做 1x1 融合（不改通道数，仍为 3*branch_out）
        self.fuse_pw = nn.Sequential(
            nn.Conv2d(3 * branch_out, 3 * branch_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(3 * branch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B, 1, C, T]

        feats = []
        for conv, bn in zip(self.convs, self.bns):
            y = conv(x)                 # [B, branch_out, C, T]
            y = bn(y)
            y = F.relu(y, inplace=True)
            feats.append(y)             # [B, branch_out, C, T]

        y = torch.cat(feats, dim=1)     # [B, 3*branch_out, C, T]
        y = self.fuse_pw(y)             # [B, 3*branch_out, C, T]
        return y


class SpatialConvBlock(nn.Module):
    """
    跨导联的 depthwise 卷积（核=C×1），再降通道、池化：
      输入:  [B, F_temporal, C, T]
      输出:  [B, F_spatial(=16), 1, T/4]
    """
    def __init__(self, n_channels: int, in_feat: int = 24, out_feat: int = 16,
                 branch_out: int = 8, p_drop: float = 0.25):
        super().__init__()
        self.depthwise_spatial = nn.Conv2d(
            in_channels=in_feat, out_channels=out_feat,
            kernel_size=(n_channels, 1), groups=branch_out, bias=False
        )
        self.bn = nn.BatchNorm2d(out_feat)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        x = self.depthwise_spatial(x)   # [B, out_feat, 1, T]
        x = self.act(self.bn(x))
        x = self.pool(x)                # [B, 16, 1, T/4]
        x = self.drop(x)
        return x


class SeparableConv(nn.Module):
    """标准可分离卷积: depthwise(1×k) + pointwise(1×1)"""
    def __init__(self, channels: int, k: int):
        super().__init__()
        pad = k // 2
        self.depth = nn.Conv2d(channels, channels, kernel_size=(1, k),
                               padding=(0, pad), groups=channels, bias=False)
        self.point = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.point(self.depth(x))


class FeatureFusionConvBlock(nn.Module):
    """
    可分离卷积 + BN + ReLU + AvgPool(×1/8) + Dropout
      输入:  [B, 16, 1, T/4]
      输出:  [B, 16, 1, T/32]
    """
    def __init__(self, k_sep: int = 7, p_drop: float = 0.25):
        super().__init__()
        self.sep = SeparableConv(16, k_sep)
        self.bn  = nn.BatchNorm2d(16)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        x = self.sep(x)
        x = self.act(self.bn(x))
        x = self.pool(x)
        x = self.drop(x)
        return x


class ClassificationConvBlock(nn.Module):
    """
    1×1 pointwise -> GlobalAvgPool -> logits
      输入:  [B, 16, 1, T/32]
      输出:  logits [B, n_class], feat [B, 16]
    """
    def __init__(self, n_class: int):
        super().__init__()
        self.head = nn.Conv2d(16, n_class, kernel_size=1, bias=True)

    def forward(self, x):
        # 分类 head + GAP
        logits_map = self.head(x)                                        # [B, n_class, 1, T/32]
        logits = F.adaptive_avg_pool2d(logits_map, output_size=(1, 1)).flatten(1)  # [B, n_class]
        return logits


class Model(nn.Module):
    """
    LENet (Lightweight & Efficient Net) 复现
    输入:  x [B,1,C,T]
    输出:  (logits, feat)  其中 feat 为分类前 GAP 的 16-维表征 (L2-norm)
    """
    def __init__(self, config: dict):
        super().__init__()
        self.n_channels = int(config["n_channels"])
        self.fs         = int(config["fs"])
        self.n_class    = int(config["n_class"])

        # --- Block 1: Temporal ---
        branch_out   = int(config.get("temporal_branch_out", 8))
        k_bases      = tuple(config.get("temporal_k_bases", [16, 32, 64]))
        fs_ref       = int(config.get("temporal_fs_ref", 256))
        self.temporal = TemporalConvBlock(1, self.fs, branch_out, k_bases, fs_ref)  # -> [B, 24, C, T]

        # --- Block 2: Spatial ---
        self.spatial = SpatialConvBlock(self.n_channels, in_feat=3*branch_out, out_feat=16,
                                        branch_out=branch_out, p_drop=float(config.get("spatial_dropout", 0.25)))       # -> [B,16,1,T/4]

        # --- Block 3: Feature Fusion ---
        k_sep = int(config.get("fusion_k", 7))
        self.fusion = FeatureFusionConvBlock(k_sep=k_sep, p_drop=float(config.get("fusion_dropout", 0.25)))  # -> [B,16,1,T/32]

        # --- Block 4: Classification ---
        self.classifier = ClassificationConvBlock(self.n_class)

    def forward(self, x, train_stage: int = 2):
        # x: [B,1,C,T]
        x = self.temporal(x)             # [B,24,C,T]
        x = self.spatial(x)              # [B,16,1,T/4]
        x = self.fusion(x)               # [B,16,1,T/32]
        logits = self.classifier(x)
        return logits



if __name__ == "__main__":
    # 一个可运行的小配置
    args = {
        "n_channels": 62,
        "fs": 128,
        "n_class": 2,
        "seq_len": 128,
        # Block1
        "temporal_branch_out": 8,
        "temporal_k_bases": [16, 32, 64],
        "temporal_fs_ref": 256,
        # Block2/3
        "spatial_dropout": 0.25,
        "fusion_k": 7,
        "fusion_dropout": 0.25,
    }
    model = Model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=(1, 1, args["n_channels"], args["seq_len"]))
