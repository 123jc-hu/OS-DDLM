from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from torch import nn
from models.EEGNet import Conv2dWithConstraint, DenseWithConstraint, calculate_outsize
from einops.layers.torch import Rearrange
import torch
from torchinfo import summary
from utils.config import load_config


class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_channels = config["n_channels"]
        self.fs = config["fs"]
        self.n_classes = config["n_class"]

        self.BasicBlock = self.feature_extract_blocks()
        self.BasicBlockOutputSize = calculate_outsize(self.BasicBlock, self.n_channels, self.fs)
        self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize)

    def feature_extract_blocks(self):
        Block1 = nn.Sequential(
            Conv2dWithConstraint(
                1,
                8,
                (1, self.fs//4),
                max_norm=0.5,
                stride=(1, self.fs//32),
                bias=False,
            ),
            nn.BatchNorm2d(8),
        )

        Block2 = nn.Sequential(
            Rearrange("b k c t -> b t c k"),
            Conv2dWithConstraint(
                25,
                25,
                (self.n_channels, 1),
                max_norm=0.5,
                stride=(1, 1),
                bias=False,
                groups=25,
            ),
            Rearrange("b t c k -> b k c t"),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout2d(p=0.25)
        )

        Block3 = nn.Sequential(
            Conv2dWithConstraint(
                8,
                8,
                (1, 9),
                max_norm=0.5,
                stride=(1, 1),
                bias=False,
                groups=8,
            ),
            Conv2dWithConstraint(
                8,
                16,
                1,
                stride=(1, 1),
                bias=False,
                max_norm=0.5
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 17)),
            nn.Dropout2d(p=0.5)
        )
        return nn.Sequential(Block1, Block2, Block3)

    def classifier_block(self, input_size):
        Block4 = nn.Sequential(
            nn.Flatten(),
            DenseWithConstraint(
                input_size,
                self.n_classes,
                bias=False,
                max_norm=0.1
            ),
            # nn.Softmax(dim=1)
        )
        return Block4

    def forward(self, x, stage=1, mask=None, **kwargs):
        x = self.BasicBlock(x)
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
