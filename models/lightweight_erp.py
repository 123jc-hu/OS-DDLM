from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from utils.config import load_config


# -------------------------
# Utilities
# -------------------------
def _odd(k: int) -> int:
    """force odd kernel length for 'same' padding semantic"""
    return k if k % 2 == 1 else k + 1


# =========================
# SimAM (parameter-free attention)
# =========================
class SimAM(nn.Module):
    def __init__(self, lambda_val: float = 1e-4):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T) or (B,C,H,W)
        if x.dim() == 3:
            x_ = x.unsqueeze(2)  # (B,C,1,T)
        elif x.dim() == 4:
            x_ = x
        else:
            raise ValueError("SimAM expects 3D/4D input")

        mu = x_.mean(dim=(2, 3), keepdim=True)
        var = ((x_ - mu) ** 2).mean(dim=(2, 3), keepdim=True)
        e = ((x_ - mu) ** 2) / (2 * var + 1e-12) + self.lambda_val
        attn = torch.sigmoid(1.0 / (e + 1e-12))
        out = attn * x_
        return out.squeeze(2) if x.dim() == 3 else out


# =========================
# True Depthwise Deformable Conv1d (per 2.8)
# =========================
class DepthwiseDeformConv1d(nn.Module):
    """
    Depthwise deformable 1D conv implemented via grid_sample.

    - channels: number of input/output channels (depthwise)
    - kernel_size: K
    - padding: 'valid' (as in Table 2) or 'same'
    - stride: only 1 is supported here (paper没强调 stride，表格也是 stride=1)
    - bias: False（保持简洁；若需要可自行加上逐通道 bias）
    """
    def __init__(self, channels: int, kernel_size: int, padding: str = "valid"):
        super().__init__()
        assert padding in ("valid", "same")
        self.channels = channels
        self.kernel_size = int(kernel_size)
        self.padding = padding

        # depthwise conv weights: (C, K)
        self.weight = nn.Parameter(torch.empty(channels, self.kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # offset predictor: depthwise Conv1d, 输出 C*K 个 offset
        # 按论文做法，我们用与主卷积相同的 K 和 depthwise 分组
        pad = 0 if padding == "valid" else self.kernel_size // 2
        self.offset_conv = nn.Conv1d(
            channels, channels * self.kernel_size,
            kernel_size=self.kernel_size, padding=pad,
            groups=channels, bias=True
        )
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        return: (B, C, T_out)
        """
        B, C, T = x.shape
        assert C == self.channels
        K = self.kernel_size

        # output temporal length（与常规 conv 对齐）
        if self.padding == "valid":
            T_out = T - K + 1
            pad_in = 0
        else:  # 'same'
            T_out = T
            pad_in = K // 2

        # 预测 offsets: (B, C*K, T_out) -> (B, C, K, T_out)
        offsets = self.offset_conv(x)  # 这里内部已做好 padding 以匹配 T_out
        offsets = offsets.view(B, C, K, -1)
        assert offsets.shape[-1] == T_out

        # 为 grid_sample 构造输入: 把 (B,C,T) 视作 (B*C, 1, 1, T)
        inp = x.view(B * C, 1, 1, T)

        # 归一化系数（align_corners=True 时用 T-1）
        denom = max(T - 1, 1)

        # 累加输出
        y = x.new_zeros(B, C, T_out)

        # 对每个核元素 n，采样位置 base+n+Δn，并双线性插值
        # base 索引：valid 时 [0..T_out-1]；same 时 [-pad..T-pad-1]，但我们已在 offset 分支内部做了 padding，
        # 这里依然以 0..T_out-1 作为输出步进，并用 n 偏移对齐 receptive field。
        base = torch.arange(T_out, device=x.device).view(1, 1, T_out)  # (1,1,T_out)

        for n in range(K):
            # 绝对采样位置（以样本点为单位）
            pos = base + n - (0 if self.padding == "valid" else K // 2) \
                  + offsets[:, :, n, :]  # (B,C,T_out)

            # 归一化到 [-1,1] -> grid_sample 的 W 方向（x）
            x_norm = 2.0 * (pos / denom) - 1.0
            y_norm = torch.zeros_like(x_norm)  # H=1 平面，y 恒为 0

            grid = torch.stack([x_norm, y_norm], dim=-1)          # (B,C,T_out,2)
            grid = grid.view(B * C, 1, T_out, 2)                   # (B*C,1,T_out,2)

            # 采样：在宽度维（时间）插值
            sampled = F.grid_sample(
                inp, grid, mode="bilinear", padding_mode="zeros",
                align_corners=True
            )  # (B*C,1,1,T_out)

            sampled = sampled.view(B, C, T_out)                    # (B,C,T_out)

            # 按 depthwise 权重累加
            w_n = self.weight[:, n].view(1, C, 1)                  # (1,C,1)
            y = y + sampled * w_n

        return y  # (B,C,T_out)


# =========================
# Proposed Model (含真实DeformConv1d)
# =========================
def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


class Model(nn.Module):
    """
    结构：Spatio-Temporal -> CompExt(三分支, 先C后T, MixConv) ->
         Pointwise+SimAM -> Advanced Temporal (两层DepthwiseDeformConv1d, valid) ->
         Dropout -> AdaptiveAvgPool1d(25) -> FC
    输入 (B,1,C,T)
    """
    def __init__(self, config: dict):
        super().__init__()
        self.n_channels = config["n_channels"]
        self.fs = config["fs"]
        self.n_class = config["n_class"]

        # ---- 1) Spatio-Temporal
        self.stem = nn.Sequential(
            nn.Conv2d(1, 15, kernel_size=(2, 3), stride=(1, 1), padding='same', bias=False),
            nn.BatchNorm2d(15),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
        )

        # ---- 2) CompExt: 三分支 (60/120/250 ms)
        k_t1 = _odd(max(3, int(round(self.fs * 0.06))))   # ≈ fs/16
        k_t2 = _odd(max(3, int(round(self.fs * 0.12))))   # ≈ fs/8
        k_t3 = _odd(max(3, int(round(self.fs * 0.25))))   # ≈ fs/4

        self.branch1 = self._make_branch(temporal_k=k_t1)
        self.branch2 = self._make_branch(temporal_k=k_t2)
        self.branch3 = self._make_branch(temporal_k=k_t3)

        self.post_concat = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 40)),  # -> (B,15,1,40)
            nn.Dropout(0.65),
            nn.Conv2d(15, 5, kernel_size=(1, 1), bias=False),  # pointwise 1x1
            nn.BatchNorm2d(5),
            nn.ELU(inplace=True),
            SimAM(),
        )

        # ---- 3) Advanced Temporal：两层真正的 Depthwise DeformConv1d（valid）
        self.def1 = DepthwiseDeformConv1d(channels=5, kernel_size=5, padding="valid")
        self.def2 = DepthwiseDeformConv1d(channels=5, kernel_size=10, padding="valid")
        self.drop_adv = nn.Dropout(0.5)
        self.pool_adv = nn.AdaptiveAvgPool1d(25)  # -> (B,5,25)

        # ---- 4) Classifier
        self.classifier = nn.Linear(5 * 25, self.n_class)

    def _make_branch(self, temporal_k: int) -> nn.Module:
        # 单支处理 5 个通道（15 被均分为 3 组）
        return nn.Sequential(
            # Spatial depthwise: kernel (C,1), valid
            nn.Conv2d(5, 5, kernel_size=(self.n_channels, 1),
                      padding=(0, 0), groups=5, bias=False),
            nn.BatchNorm2d(5),
            nn.ELU(inplace=True),
            SimAM(),
            # Temporal depthwise: kernel (1, k_t), same
            nn.Conv2d(5, 5, kernel_size=(1, temporal_k),
                      padding=(0, temporal_k // 2), groups=5, bias=False),
            nn.BatchNorm2d(5),
            nn.ELU(inplace=True),
            SimAM(),
        )

    def forward(self, x, train_stage: int = 2):
        # x: (B,1,C,T)
        x = self.stem(x)                  # (B,15,C,T)

        # 三分组（MixConv 思想：把 15 个通道分成 3 组，每组 5 个）
        x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)  # (B,5,C,T) each
        b1 = self.branch1(x1)   # (B,5,1,T)
        b2 = self.branch2(x2)   # (B,5,1,T)
        b3 = self.branch3(x3)   # (B,5,1,T)

        x = torch.cat([b1, b2, b3], dim=1)   # (B,15,1,T)
        x = self.post_concat(x)              # (B,5,1,40)

        x = x.squeeze(2)                     # (B,5,40)
        x = self.def1(x)                     # (B,5,36)
        x = self.def2(x)                     # (B,5,27)
        x = self.drop_adv(x)
        x = self.pool_adv(x)                 # (B,5,25)

        x = x.flatten(1)                     # (B,125)
        return self.classifier(x), None


if __name__ == '__main__':
    config_path = "config.yaml"
    args = load_config(config_path)
    model = Model(args)
    input_shape = (1, 1, 62, 128)  # 输入形状 (batch_size, n_channels, seq_len)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用 torchsummary 打印模型信息
    summary(model, input_shape)

    # 手动测试模型
    # x = torch.randn(1, 1, 62, 128).to(device)  # 输入数据
    # output = model(x)
    # print(output[0].shape)  # 检查输出形状