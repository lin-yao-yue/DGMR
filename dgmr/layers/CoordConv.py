import torch
import torch.nn as nn

"""
https://blog.csdn.net/oYeZhou/article/details/116717210

传统卷积具备平移不变性，这使得其在应对分类等任务时可以更好的学习本质特征。
不过，当需要感知位置信息时，传统卷积就有点力不从心了。即传统卷积没有空间感知能力
传统卷积在卷积核进行局部运算时，是不知道当前卷积核所处的空间位置的，相当于盲人摸象，能感受到局部信息，但对这部分信息在整体的哪个位置没有概念。
为了使得卷积能够感知空间信息，在输入feature map后面增加了两个coordinate通道，分别表示原始输入的x和y坐标，然后再进行传统卷积，从而使得卷积过程可以感知feature map的空间信息，
该方法称之为CoordConv

"""


# 对输入的张量进行处理
class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        # torch.repeat: 沿着给定的维度对tensor进行重复，1表示不变
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        # 在 channel 维度进行拼接
        ret = torch.cat(
            [input_tensor, xx_channel.type_as(input_tensor), yy_channel.type_as(input_tensor)],
            dim=1,
        )

        # torch.pow(input, exp)：将input作为底数，exp作为指数，exp也可以是张量
        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2)
                + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2)
            )
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        # 添加两个通道来识别过滤器的位置
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        # 对张量进行处理
        ret = self.addcoords(x)
        # 将该张量放入卷积层中
        ret = self.conv(ret)
        return ret
