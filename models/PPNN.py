from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn.functional as F
import torch.nn as nn
from models.EEGNet import calculate_outsize
from torchinfo import summary
from utils.config import load_config


class Model(nn.Module):
    def __init__(self, configs: dict):
        super(Model, self).__init__()
        self.n_channels = configs["n_channels"]
        self.fs = configs["fs"]

        # Block1: 5层dilated conv2d，每层dilated输出维度都是8，kernal是(1,3)，padding=same，dilation分别是(1,2),(1,4),(1,8),(1,16),(1,32)
        self.Block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 3), padding=(0, 2), dilation=(1, 2)),
            nn.Conv2d(8, 8, kernel_size=(1, 3), padding=(0, 4), dilation=(1, 4)),
            nn.Conv2d(8, 8, kernel_size=(1, 3), padding=(0, 8), dilation=(1, 8)),
            nn.Conv2d(8, 8, kernel_size=(1, 3), padding=(0, 16), dilation=(1, 16)),
            nn.Conv2d(8, 8, kernel_size=(1, 3), padding=(0, 32), dilation=(1, 32)),
            nn.BatchNorm2d(8),
            nn.ELU()
        )

        # Block2: 一层conv2d，输出维度是16，kernal是(C, 1)，没有padding，然后接BN层、ELU和dropout
        self.Block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(self.n_channels, 1)),  # input_channels 表示通道数C
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        self.BasicBlockOutputSize = calculate_outsize(
            nn.Sequential(self.Block1, self.Block2), self.n_channels, self.fs)

        # 分类层: Flatten+全连接层，输出为2
        self.ClassifierBlock = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.BasicBlockOutputSize, 2)
        )

    def forward(self, x, train_stage: int = 2):
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.ClassifierBlock(x)
        return x


if __name__ == '__main__':
    config_path = "config.yaml"
    args = load_config(config_path)
    model = Model(args)
    input_shape = (1, 1, 62, 128)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用 summary 打印模型信息
    summary(model, input_shape, device=str(device))

    # 手动测试模型
    # x = torch.randn(1, 10, 62, 128).to(device)  # 输入数据
    # output = model(x)
    # print(output[0].shape)  # 检查输出形状
