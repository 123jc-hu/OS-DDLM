from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from utils.config import load_config


def _scale_kernel(base_k: int, fs: int, fs_ref: int = 128, min_k: int = 3) -> int:
    """
    将时间卷积核按照采样率缩放（表里128点窗口时用32）。
    这样在不同 fs 下保持大致时间感受野一致；若你想完全固定为32，可把 fs_ref=fs。
    """
    k = max(min_k, int(round(base_k * fs / fs_ref)))
    return k


class Model(nn.Module):
    """
    1DCNN-BiLSTM
    输入:  x [B, 1, C, T]
    输出:  (logits, feat)  其中 feat 是 LSTM 最后一层双向隐藏态拼接后的 32 维向量的 L2 归一化
    """
    def __init__(self, config: dict):
        super().__init__()
        self.n_channels = int(config["n_channels"])   # 例如 32
        self.fs         = int(config["fs"])           # 例如 128
        self.n_class    = int(config["n_class"])      # 例如 2

        # ------- 1D CNN (表：Conv1d -> BN1 -> MaxPool) -------
        # 表中参数：kernel size=32, stride=1, out_channels=8, in_channels=32 → 8200 参数
        # 这里默认按 fs_ref=128 缩放；若希望严格固定为 32，可改 fs_ref=self.fs
        k = _scale_kernel(base_k=32, fs=self.fs, fs_ref=128)
        self.conv1 = nn.Conv1d(
            in_channels=self.n_channels, out_channels=8,
            kernel_size=k, stride=1, padding=0, bias=True
        )
        self.bn1   = nn.BatchNorm1d(8)
        # 表中 MaxPool 输出 48，说明 kernel=2, stride=2
        self.pool  = nn.MaxPool1d(kernel_size=2, stride=2)

        # ------- BiLSTM (hidden_size=16, 双向 → 输出维 32) -------
        # LSTM 输入特征维 = 8（来自 conv 通道数）
        self.lstm = nn.LSTM(
            input_size=8, hidden_size=16, num_layers=1,
            batch_first=True, bidirectional=True
        )
        self.bn2 = nn.BatchNorm1d(32)

        # ------- 分类头 -------
        self.fc  = nn.Linear(32, self.n_class)

    def forward(self, x, train_stage: int = 2):
        """
        x: [B, 1, C, T]
        return: logits [B, n_class], feat [B, 32] (L2-normalized)
        """
        B = x.size(0)
        # 1D CNN 分支：把 [B,1,C,T] 变为 [B,C,T] 再做 Conv1d
        x = x.squeeze(1)                          # [B, C, T]
        x = self.conv1(x)                         # [B, 8, 97] (当 T=128 且 k=32 时)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)                          # [B, 8, 48] (当上面为 97 时)

        # 准备给 LSTM： [B, 8, L] -> [B, L, 8]
        x = x.transpose(1, 2)                     # [B, L, 8]
        # BiLSTM
        _, (h_n, _) = self.lstm(x)                # h_n: [num_layers*2, B, 16]
        # 取最后一层双向隐藏态并拼接 → [B, 32]
        h_fwd = h_n[-2]                           # [B, 16]
        h_bwd = h_n[-1]                           # [B, 16]
        feat = torch.cat([h_fwd, h_bwd], dim=1)   # [B, 32]
        feat = self.bn2(feat)
        logits = self.fc(feat)                    # [B, n_class]

        feat = F.normalize(feat, p=2, dim=-1)     # 与你的评估/对齐流程一致
        return logits, feat


if __name__ == "__main__":
    # 仅用于快速检查
    cfg = {"n_channels": 62, "fs": 128, "n_class": 2, "seq_len": 128}
    model = Model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=(1, 1, cfg["n_channels"], cfg["seq_len"]))
