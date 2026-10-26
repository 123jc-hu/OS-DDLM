from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from torch import nn
from einops.layers.torch import Rearrange, Reduce
from models.EEGNet import calculate_outsize
import torch
from torchinfo import summary
from utils.config import load_config
import torch.nn.functional as F
from models.Ours_4 import LinearWithConstraint
from torch.autograd import Function


class SpatialFusionLowRank(nn.Module):
    def __init__(self, n_channels, group_dim=4, r=8):  # r=6~8 一个甜点
        super().__init__()
        self.proj = nn.Conv2d(group_dim, r, kernel_size=(n_channels, 1), bias=False)   # C->r
        self.mix  = nn.Conv2d(r, 16, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(16)
        self.act  = nn.ReLU()

    def forward(self, x):  # x: [B, (c*h)=C*4, 1, T'] 已经rearrange过
        # 你原本的 Rearrange 先做：
        # x: [B, (c h) k t] -> [B, h, c, (k t)], 其中 h=4
        x = self.proj(x)   # [B, r, 1, T']
        x = self.mix(x)    # [B, 16, 1, T']
        return self.act(self.bn(x))
    

class ECA(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
    def forward(self, x):  # [B, C, H, W]
        y = self.pool(x)                  # [B,C,1,1]
        y = self.conv(y.squeeze(-1).transpose(1,2)).transpose(1,2).unsqueeze(-1)
        return x * torch.sigmoid(y)


class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_channels = config["n_channels"]
        self.fs = config["fs"]
        self.n_class = config["n_class"]
        group_dim = config.get("group_dim", 4)
        k_small = 9
        d = max(1, (self.fs//4 - 1) // (k_small - 1))  # 近似匹配 RF
        pad = d * (k_small // 2)

        self.channel_tcb_block = nn.Sequential(
            Rearrange("b k c t -> b c k t"),
            nn.Conv2d(in_channels=self.n_channels, out_channels=self.n_channels*group_dim, kernel_size=(1, k_small), stride=(1, 2),
                padding=(0, pad), dilation=(1, d), groups=self.n_channels, bias=False),
            nn.BatchNorm2d(self.n_channels * group_dim),
            nn.ELU(),
        )

        self.spatiotemporal_fusion_block = nn.Sequential(
            Rearrange("b (c h) k t -> b h c (k t)", h=group_dim),
            SpatialFusionLowRank(self.n_channels, group_dim=group_dim, r=4),
            ECA(16, k=3),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
        )

        self.output_size = 16 * (self.fs // 8)

        self.primary_classifier = LinearWithConstraint(self.output_size, self.n_class, max_norm=0.5, doWeightNorm=True)

    def forward(self, x, train_stage=2):
        # x: [B, 1, C, T]
        x = self.channel_tcb_block(x)
        x = self.spatiotemporal_fusion_block(x)

        feat = F.normalize(x.flatten(start_dim=1), p=2, dim=-1)
        logits = self.primary_classifier(x.flatten(start_dim=1))

        return logits, feat


if __name__ == '__main__':
    args = {"n_channels": 48, "fs": 128, "n_class": 2, "seq_len": 128}
    model = Model(args)
    input_shape = (1, 1, args["n_channels"], args["seq_len"])  # 输入形状 (batch_size, n_channels, seq_len)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用 torchsummary 打印模型信息
    summary(model, input_shape)

    # 手动测试模型
    # x = torch.randn(1, 1, 62, 128).to(device)  # 输入数据
    # output = model(x)
    # print(output[0].shape)  # 检查输出形状