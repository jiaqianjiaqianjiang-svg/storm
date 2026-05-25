"""风暴潮重建 CNN 模型。

本项目的预处理阶段会把某一天 D 的气象场组织为：

- D-1 和 D 两天；
- ERA20C 每 3 小时一个时间片，两天共 16 个时间片；
- U10、V10、SLP 三个变量；
- 每个时间片插值到 40 x 40 网格。

因此单个样本的输入形状是 ``(48, 40, 40)``。这里的 CNN 第一层直接接收
48 个通道，输出当天的 daily maximum storm surge 标准化值。
"""

from __future__ import annotations

import torch
from torch import nn


class StormSurgeCNN(nn.Module):
    """适配 ``(48, 40, 40)`` 输入的轻量 CNN。

    这个结构不是为了堆很深，而是先稳定复现论文里的核心思路：用站点周围
    10° x 10° 的 U10/V10/SLP 气象场预测逐日最大风暴潮。1985 单年样本很少，
    所以模型容量刻意保持克制，并加入 Dropout 减少过拟合。
    """

    def __init__(self, in_channels: int = 48, dropout: float = 0.30) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 40 -> 20

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 20 -> 10

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 10 -> 5
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数
        ----
        x:
            形状为 ``(batch, 48, 40, 40)`` 的张量。

        返回
        ----
        torch.Tensor
            形状为 ``(batch,)`` 的预测值，单位是标准化后的 storm surge。
        """

        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(-1)


def count_trainable_parameters(model: nn.Module) -> int:
    """统计模型中需要训练的参数量，便于日志检查模型规模。"""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
