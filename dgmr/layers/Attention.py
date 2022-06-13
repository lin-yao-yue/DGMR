import torch
import torch.nn as nn
from torch.nn import functional as F
import einops


# 根据 张量某通道的 query key value 矩阵，求出该通道的 attention 值
def attention_einsum(q, k, v):
    """Apply the attention operator to tensors of shape [h, w, c]."""

    # Reshape 3D tensors to 2D tensor with first dimension L = h x w.
    """
    rearrange按给出的模式(注释)重组张量
    """
    k = einops.rearrange(k, "h w c -> (h w) c")  # [h, w, c] -> [L, c]
    v = einops.rearrange(v, "h w c -> (h w) c")  # [h, w, c] -> [L, c]

    # Einstein summation corresponding to the query * key operation.
    """
    F.softmax：根据指定维度dim将张量的每个元素缩放到（0,1）区间且和为1
    dim：0表示张量的最高维度，1表示张张量的次高维度，2表示张量的次次高维度，以此类推。-1表示张量维度的最低维度，-2表示倒数第二维度，-3表示倒数第三维度。
    
    torch.einsum：根据指定模式进行矩阵运算
    """

    # 根据Query和Key计算权重系数，归一化的目的就是算出每一个对应的 value 所拥有的权重
    beta = F.softmax(torch.einsum("hwc, Lc->hwL", q, k), dim=-1)

    # Einstein summation corresponding to the attention * value operation.
    # 根据权重系数对Value进行加权求和，得到Attention数值.
    out = torch.einsum("hwL, Lc->hwc", beta, v)
    return out


"""
总体为残差块
计算 X 每一个通道的 attention 值（即空间attention），在此基础上放入一个卷积层中 得到 Y
return X+Y
"""


class AttentionLayer(torch.nn.Module):
    """Attention Module"""

    def __init__(self, input_channels: int, output_channels: int, ratio_kq=8, ratio_v=8):
        super(AttentionLayer, self).__init__()

        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v
        self.output_channels = output_channels
        self.input_channels = input_channels

        # Compute query, key and value using 1x1 convolutions.
        # 计算query矩阵的卷积层
        self.query = torch.nn.Conv2d(
            in_channels=input_channels,
            # 地板除即向下取整
            out_channels=self.output_channels // self.ratio_kq,
            kernel_size=(1, 1),
            # valid 方式的填充表示不能再兼容下一个步长时就舍弃
            # SAME 则表示会用填充去满足卷积核下一个步长的运算
            padding="valid",
            bias=False,
        )
        self.key = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.output_channels // self.ratio_kq,
            kernel_size=(1, 1),
            padding="valid",
            bias=False,
        )
        self.value = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.output_channels // self.ratio_v,
            kernel_size=(1, 1),
            padding="valid",
            bias=False,
        )
        self.last_conv = torch.nn.Conv2d(
            in_channels=self.output_channels // 8,
            out_channels=self.output_channels,
            kernel_size=(1, 1),
            padding="valid",
            bias=False,
        )

        # Learnable gain parameter
        # 一个要通过学习得到的参数
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute query, key and value using 1x1 convolutions.
        # x的通道数为output_channels/8，即以下三个变量通过卷积层运算后通道数与x相同
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Apply the attention operation.
        out = []
        # 遍历batch
        for b in range(x.shape[0]):
            # Apply to each in batch
            # 求出每一个batch的 attention 值
            out.append(attention_einsum(query[b], key[b], value[b]))
        # torch.stack：沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
        out = torch.stack(out, dim=0)
        out = self.gamma * self.last_conv(out)
        # Residual connection.
        return out + x
