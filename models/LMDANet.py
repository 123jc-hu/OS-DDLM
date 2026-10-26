from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from utils.config import load_config


class EEGDepthAttention(nn.Module):
    """
    EEG Depth Attention Module
    输入: x [B, depth(D), channels(C), time(T)]
    """
    def __init__(self, T, depth, k=7):
        super().__init__()
        self.depth = depth
        # Semi-global pooling: 在 EEG 通道维(C)做全局平均池化，保留时间维
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, T))
        # Local cross-depth interaction: 仅在 depth 维做卷积筛选
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)
        self.softmax = nn.Softmax(dim=2)  # 在 depth 维做 softmax

    def forward(self, x):
        # [B, D, C, T]
        x_pool = self.adaptive_pool(x)          # [B, D, 1, T]
        x_transpose = x_pool.transpose(1, 2)    # [B, 1, D, T]
        y = self.conv(x_transpose)              # [B, 1, D, T]
        y = self.softmax(y)                     # [B, 1, D, T]
        y = y.transpose(1, 2)                   # [B, D, 1, T]
        return y * self.depth * x


class Model(nn.Module):
    """
    统一到你的模型风格：
    - __init__(config)
        * 读取: n_channels, fs, n_class
        * 可选: seq_len（窗口长度，默认为fs或128），以及LMDA相关超参
    - forward(x, train_stage=2) -> (logits, feat)
        * x: [B, 1, C, T]
        * logits: 分类器输出
    网络主体严格按原LMDA实现，不做结构改动。
    """
    def __init__(self, config: dict):
        super().__init__()
        # 基础配置
        self.n_channels = int(config["n_channels"])
        self.fs = int(config["fs"])
        self.n_class = int(config["n_class"])

        # 窗口长度（LMDA原实现需要 samples 参与确定分类器in_features）
        # 若你的数据窗口 T 与 fs 不同，可在config里显式给 seq_len；否则退化为常见的128或fs
        self.seq_len = int(config.get("seq_len", config.get("window_len", 128 if self.fs <= 128 else self.fs)))

        # LMDA 原超参（保持默认不变，可在config里覆盖）
        depth = int(config.get("lmda_depth", 9))
        kernel = int(config.get("lmda_kernel", 39))
        channel_depth1 = int(config.get("lmda_channel_depth1", 24))
        channel_depth2 = int(config.get("lmda_channel_depth2", 9))
        avgpool = int(config.get("lmda_avgpool", 1))

        # ====== 按原代码构建模块（不改结构/超参默认值） ======
        # 可学习的“导联权重筛选”
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, self.n_channels), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        # 时域块
        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel), groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )

        # 通道卷积块
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(self.n_channels, 1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        # 归一化/池化 + dropout
        self.norm = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, avgpool)),
            nn.Dropout(p=0.65),
        )

        # ====== 依据样本长度推断classifier的输入维度（与原版一致的“dummy forward”做法） ======
        with torch.no_grad():
            dummy = torch.ones((1, 1, self.n_channels, self.seq_len))
            out = torch.einsum('bdcw, hdc->bhcw', dummy, self.channel_weight)  # [B, depth, C, T] -> [B, H=depth, C, T]
            out = self.time_conv(out)      # [B, channel_depth1, C, T']

            # 深度注意力需要 W 和 C（这里Ctmp=channel_depth1，Htmp=C）
            self.depthAttention = EEGDepthAttention(self.fs-kernel+1, channel_depth1, k=7)

            out = self.depthAttention(out) # 与原前向一致：先DA
            out = self.chanel_conv(out)    # [B, channel_depth2, 1, T'']
            out = self.norm(out)           # [B, channel_depth2, 1, T''/avepool]

            n_out_time = out.shape  # (1, channel_depth2, 1, T_final)
            flat_dim = int(n_out_time[-1] * n_out_time[-2] * n_out_time[-3])

        self.classifier = nn.Linear(flat_dim, self.n_class)

    def forward(self, x, train_stage: int = 2):
        """
        x: [B, 1, C, T]
        return: logits
        """
        # 导联权重筛选（保持原einsum写法与形状）
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)   # [B, depth, C, T]

        x_time = self.time_conv(x)            # [B, channel_depth1, C, T']
        x_time = self.depthAttention(x_time)  # DA

        x = self.chanel_conv(x_time)          # [B, channel_depth2, 1, T'']
        x = self.norm(x)                      # [B, channel_depth2, 1, T_final]

        features = torch.flatten(x, 1)        # 展平
        logits = self.classifier(features)
        return logits


if __name__ == '__main__':
    config_path = "config.yaml"
    args = load_config(config_path)

    # 关键：需在config里提供 n_channels, fs, n_class；可选 seq_len（窗口长度）
    # 若未提供 seq_len，则默认使用 min(128, fs或fs本身)，与常见窗口一致
    model = Model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 用 config 的窗口长度作为 T 来打印 summary
    T = int(args.get("seq_len", args.get("window_len", 128 if args["fs"] <= 128 else args["fs"])))
    input_shape = (1, 1, int(args["n_channels"]), T)
    summary(model, input_size=input_shape)
