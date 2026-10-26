from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from utils.config import load_config
from utils.utils import compute_hilbert_components


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1., **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(input)


class DenseWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1., **kwargs):
        self.max_norm = max_norm
        super(DenseWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(DenseWithConstraint, self).forward(input)


def calculate_outsize(model, channels, samples):
    """
    Calculate the output based on input size.
    model is from nn.Module and inputSize is an array.
    """
    data = torch.rand(1, 1, channels, samples)
    model.eval()
    out = model(data).shape
    return out.numel()


class Model(nn.Module):
    def __init__(self, config: dict):
        super(Model, self).__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.n_channels = config["n_channels"]
        self.fs = config["fs"]
        self.n_classes = config["n_class"]
        self.kernel_length = self.fs // 2
        self.kernel_length2 = self.fs // 8
        self.dropout_rate = 0.5

        self.BasicBlock = self.feature_extract_blocks()
        self.BasicBlockOutputSize = calculate_outsize(self.BasicBlock, self.n_channels, self.fs)
        self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize)

    def feature_extract_blocks(self):
        Block1 = nn.Sequential(
            nn.Conv2d(
                1,
                self.F1,
                (1, self.kernel_length),
                stride=1,
                bias=False,
                padding=(0, self.kernel_length // 2),
            ),
            nn.BatchNorm2d(self.F1),
            # DepthwiseConv2D =======================
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (self.n_channels, 1),
                max_norm=1,
                stride=1,
                bias=False,
                groups=self.F1,
            ),
            # ========================================
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(p=self.dropout_rate)
        )

        Block2 = nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                (1, self.kernel_length2),
                stride=1,
                bias=False,
                padding=(0, self.kernel_length2 // 2),
                groups=self.F1 * self.D
            ),
            nn.Conv2d(
                self.F1 * self.D,
                self.F2,
                (1, 1),
                stride=1,
                bias=False,
            ),
            # ========================================
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=self.dropout_rate)
        )
        return nn.Sequential(Block1, Block2)

    def classifier_block(self, input_size):
        return nn.Sequential(
            nn.Flatten(),
            DenseWithConstraint(
                input_size,
                self.n_classes,
                bias=False,
                max_norm=0.25),
            # nn.Softmax(dim=1)
        )

    def forward(self, x, train_stage: int = 2, mask=None, **kwargs):
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
