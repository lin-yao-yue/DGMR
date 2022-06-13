from typing import Tuple

import einops
import torch
import torch.nn.functional as F
from torch.distributions import normal
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.modules.pixelshuffle import PixelUnshuffle
from dgmr.layers.utils import get_conv_layer
from dgmr.layers import AttentionLayer
from huggingface_hub import PyTorchModelHubMixin

"""
G,D,L Block 均为残差块的实现，具体见论文page18
"""


"""
该块的每一个卷积层 = 卷积运算 + 谱归一化
1，为满足返回输入与输出的加和，要先判断二者的通道数是否相同，不相同要通过 1*1 卷积层进行通道的转换
2，批量归一化---激活函数---第一个卷积层---批量归一化---激活函数---第二个卷积层
3，return X+Y
"""


class GBlock(torch.nn.Module):
    """Residual generator block without upsampling"""

    # 没有上采样的残差生成器模块
    # 上采样是将图像转为更高分辨率后采样

    def __init__(
            self,
            input_channels: int = 12,
            output_channels: int = 12,
            conv_type: str = "standard",
            spectral_normalized_eps=0.0001,
    ):
        """
        G Block from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Type of convolution desired, see satflow/models/utils.py for options
        """
        super().__init__()
        self.output_channels = output_channels
        # 批量归一化
        self.bn1 = torch.nn.BatchNorm2d(input_channels)
        self.bn2 = torch.nn.BatchNorm2d(input_channels)
        self.relu = torch.nn.ReLU()
        # 默认是普通二维卷积层：torch.nn.Conv2d
        conv2d = get_conv_layer(conv_type)
        # Upsample in the 1x1
        # 进行卷积运算后都要进行谱归一化
        self.conv_1x1 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
            ),
            # 谱归一化使用的参数
            eps=spectral_normalized_eps,
        )
        # Upsample 2D conv
        self.first_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=3,
                padding=1,
            ),
            eps=spectral_normalized_eps,
        )
        self.last_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1
            ),
            eps=spectral_normalized_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optionally spectrally normalized 1x1 convolution
        # 判断x通道数是否与输出的通道数相同，不相同的话要使用 1*1 的卷积层对x的通道数进行变换
        if x.shape[1] != self.output_channels:
            sc = self.conv_1x1(x)
        else:
            sc = x

        x2 = self.bn1(x)
        x2 = self.relu(x2)
        x2 = self.first_conv_3x3(x2)  # Make sure size is doubled
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        # Sum combine, residual connection
        # 残差网络的最后变为 X+Y
        x = x2 + sc
        return x


"""
对输入进行上采样的残差块
每一个卷积层 = 卷积运算 + 谱归一化
1，X = X---上采样---1*1卷积
2，Y = X---批量归一化---激活函数---上采样---第一个卷积层---批量归一化---激活函数---第二个卷积层
3，return X+Y
"""


class UpsampleGBlock(torch.nn.Module):
    """Residual generator block with upsampling"""
    # 进行上采样的残差生成器模块

    def __init__(
            self,
            input_channels: int = 12,
            output_channels: int = 12,
            conv_type: str = "standard",
            spectral_normalized_eps=0.0001,
    ):
        """
        G Block from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Type of convolution desired, see satflow/models/utils.py for options
        """
        super().__init__()
        self.output_channels = output_channels
        self.bn1 = torch.nn.BatchNorm2d(input_channels)
        self.bn2 = torch.nn.BatchNorm2d(input_channels)
        self.relu = torch.nn.ReLU()
        # Upsample in the 1x1
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
            ),
            eps=spectral_normalized_eps,
        )
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
        # Upsample 2D conv
        self.first_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=3,
                padding=1,
            ),
            eps=spectral_normalized_eps,
        )
        self.last_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1
            ),
            eps=spectral_normalized_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectrally normalized 1x1 convolution
        sc = self.upsample(x)
        sc = self.conv_1x1(sc)

        x2 = self.bn1(x)
        x2 = self.relu(x2)
        # Upsample
        # 上采样
        x2 = self.upsample(x2)
        x2 = self.first_conv_3x3(x2)  # Make sure size is doubled
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        # Sum combine, residual connection
        x = x2 + sc
        return x


"""
根据对于空间维度的要求（空间维度减半 or 不变）进行是否池化运算（宽和高是否缩减两倍）
1，首先要保证 输入X 输出Y 通道数相同，X = X--(conv1x1)--(pool)
2，Y = X--(RELU)--卷积层1--RELU--卷积层2--（pool）
3，return X+Y
"""


class DBlock(torch.nn.Module):
    def __init__(
            self,
            input_channels: int = 12,
            output_channels: int = 12,
            conv_type: str = "standard",
            first_relu: bool = True,
            keep_same_output: bool = False,
    ):
        """
        D and 3D Block from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Convolution type, see satflow/models/utils.py for options
            first_relu: Whether to have an ReLU before the first 3x3 convolution
            keep_same_output: Whether the output should have the same spatial dimensions as input, if False, downscales by 2
                              是否要求输出和输入有相同的空间维度，如果不是，将维度降低2倍
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.first_relu = first_relu
        self.keep_same_output = keep_same_output
        self.conv_type = conv_type
        conv2d = get_conv_layer(conv_type)
        # 根据卷积的维度选择平均池化函数
        if conv_type == "3d":
            # 3D Average pooling
            self.pooling = torch.nn.AvgPool3d(kernel_size=2, stride=2)
        else:
            self.pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_1x1 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
            )
        )
        self.first_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self.last_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            )
        )
        # Downsample at end of 3x3
        self.relu = torch.nn.ReLU()
        # Concatenate to double final channels and keep reduced spatial extent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_channels != self.output_channels:
            x1 = self.conv_1x1(x)
            # 如果不需要最后结果的空间维度与输入相同，使用池化层将宽与高的维度分别减半
            if not self.keep_same_output:
                x1 = self.pooling(x1)
        else:
            x1 = x

        if self.first_relu:
            x = self.relu(x)
        x = self.first_conv_3x3(x)
        x = self.relu(x)
        x = self.last_conv_3x3(x)

        if not self.keep_same_output:
            x = self.pooling(x)
        x = x1 + x  # Sum the outputs should be half spatial and double channels
        return x


"""
当输入通道小于输出通道时，通过对输入进行某种变换使得二者通道数相同
X = cat([X--conv_1x1, X], 1)
Y = X--relu--卷积层1--relu--卷积层2
return X+Y
"""


class LBlock(torch.nn.Module):
    """Residual block for the Latent Stack."""

    def __init__(
            self,
            input_channels: int = 12,
            output_channels: int = 12,
            kernel_size: int = 3,
            conv_type: str = "standard",
    ):
        """
        L-Block for increasing the number of channels in the input 增加输入的通道数
         from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Which type of convolution desired, see satflow/models/utils.py for options
        """
        super().__init__()
        # Output size should be channel_out - channel_in
        self.input_channels = input_channels
        self.output_channels = output_channels
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = conv2d(
            in_channels=input_channels,
            out_channels=output_channels - input_channels,
            kernel_size=1,
        )

        self.first_conv_3x3 = conv2d(
            input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=1,
            stride=1,
        )
        self.relu = torch.nn.ReLU()
        self.last_conv_3x3 = conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=1,
            stride=1,
        )

    def forward(self, x) -> torch.Tensor:

        if self.input_channels < self.output_channels:
            sc = self.conv_1x1(x)
            sc = torch.cat([x, sc], dim=1)
        else:
            sc = x

        x2 = self.relu(x)
        x2 = self.first_conv_3x3(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        return x2 + sc


"""
通过 conditioning stack 对输入的 context 进行处理
a 图左半部分的实现
"""


class ContextConditioningStack(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            input_channels: int = 1,
            output_channels: int = 768,
            num_context_steps: int = 4,
            conv_type: str = "standard",
            **kwargs
    ):
        """
        Conditioning Stack using the context images from Skillful Nowcasting, , see https://arxiv.org/pdf/2104.00954.pdf

        Args:
            input_channels: Number of input channels per timestep
            output_channels: Number of output channels for the lowest block
            conv_type: Type of 2D convolution to use, see satflow/models/utils.py for options
        """
        super().__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        output_channels = self.config["output_channels"]
        num_context_steps = self.config["num_context_steps"]
        conv_type = self.config["conv_type"]

        conv2d = get_conv_layer(conv_type)
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        # Process each observation processed separately with 4 downsample blocks
        # Concatenate across channel dimension, and for each output, 3x3 spectrally normalized convolution to reduce
        # number of channels by 2, followed by ReLU
        # 使用 4 个下采样块单独处理每个观测值
        # 跨通道尺寸串联，对于每个输出，3x3 谱归一化卷积可将通道数减少 2，然后是 ReLU

        self.d1 = DBlock(
            input_channels=4 * input_channels,
            output_channels=((output_channels // 4) * input_channels) // num_context_steps,
            conv_type=conv_type,
        )
        self.d2 = DBlock(
            input_channels=((output_channels // 4) * input_channels) // num_context_steps,
            output_channels=((output_channels // 2) * input_channels) // num_context_steps,
            conv_type=conv_type,
        )
        self.d3 = DBlock(
            input_channels=((output_channels // 2) * input_channels) // num_context_steps,
            output_channels=(output_channels * input_channels) // num_context_steps,
            conv_type=conv_type,
        )
        self.d4 = DBlock(
            input_channels=(output_channels * input_channels) // num_context_steps,
            output_channels=(output_channels * 2 * input_channels) // num_context_steps,
            conv_type=conv_type,
        )
        self.conv1 = spectral_norm(
            conv2d(
                in_channels=(output_channels // 4) * input_channels,
                out_channels=(output_channels // 8) * input_channels,
                kernel_size=3,
                padding=1,
            )
        )

        self.conv2 = spectral_norm(
            conv2d(
                in_channels=(output_channels // 2) * input_channels,
                out_channels=(output_channels // 4) * input_channels,
                kernel_size=3,
                padding=1,
            )
        )

        self.conv3 = spectral_norm(
            conv2d(
                in_channels=output_channels * input_channels,
                out_channels=(output_channels // 2) * input_channels,
                kernel_size=3,
                padding=1,
            )
        )

        self.conv4 = spectral_norm(
            conv2d(
                in_channels=output_channels * 2 * input_channels,
                out_channels=output_channels * input_channels,
                kernel_size=3,
                padding=1,
            )
        )

        self.relu = torch.nn.ReLU()

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Each timestep processed separately
        x = self.space2depth(x)
        steps = x.size(1)  # Number of timesteps
        scale_1 = []
        scale_2 = []
        scale_3 = []
        scale_4 = []
        # 对每一个 timestep 进行遍历
        for i in range(steps):
            # 依次通过DBlock
            s1 = self.d1(x[:, i, :, :, :])
            s2 = self.d2(s1)
            s3 = self.d3(s2)
            s4 = self.d4(s3)
            scale_1.append(s1)
            scale_2.append(s2)
            scale_3.append(s3)
            scale_4.append(s4)
        # 将每一个时间步的数据通过DBlock处理后再对维度进行还原
        scale_1 = torch.stack(scale_1, dim=1)  # B, T, C, H, W and want along C dimension
        scale_2 = torch.stack(scale_2, dim=1)  # B, T, C, H, W and want along C dimension
        scale_3 = torch.stack(scale_3, dim=1)  # B, T, C, H, W and want along C dimension
        scale_4 = torch.stack(scale_4, dim=1)  # B, T, C, H, W and want along C dimension
        # Mixing layer
        scale_1 = self._mixing_layer(scale_1, self.conv1)
        scale_2 = self._mixing_layer(scale_2, self.conv2)
        scale_3 = self._mixing_layer(scale_3, self.conv3)
        scale_4 = self._mixing_layer(scale_4, self.conv4)
        return scale_1, scale_2, scale_3, scale_4

    def _mixing_layer(self, inputs, conv_block):
        # Convert from [batch_size, time, h, w, c] -> [batch_size, h, w, c * time]
        # then perform convolution on the output while preserving number of c.
        stacked_inputs = einops.rearrange(inputs, "b t c h w -> b (c t) h w")
        return F.relu(conv_block(stacked_inputs))


"""
正太抽取样本后经过一系列block进行处理
b 图 LatentConditioningStack 的实现
"""


class LatentConditioningStack(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            shape: (int, int, int) = (8, 8, 8),
            output_channels: int = 768,
            use_attention: bool = True,
            **kwargs
    ):
        """
        Latent conditioning stack from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        Args:
            shape: Shape of the latent space, Should be (H/32,W/32,x) of the final image shape
            output_channels: Number of output channels for the conditioning stack
            use_attention: Whether to have a self-attention block or not
        """
        super().__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        shape = self.config["shape"]
        output_channels = self.config["output_channels"]
        use_attention = self.config["use_attention"]

        self.shape = shape
        self.use_attention = use_attention
        # 自己构造一个离散正态分布
        self.distribution = normal.Normal(loc=torch.Tensor([0.0]), scale=torch.Tensor([1.0]))

        self.conv_3x3 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=shape[0], out_channels=shape[0], kernel_size=(3, 3), padding=1
            )
        )
        self.l_block1 = LBlock(input_channels=shape[0], output_channels=output_channels // 32)
        self.l_block2 = LBlock(
            input_channels=output_channels // 32, output_channels=output_channels // 16
        )
        self.l_block3 = LBlock(
            input_channels=output_channels // 16, output_channels=output_channels // 4
        )
        if self.use_attention:
            self.att_block = AttentionLayer(
                input_channels=output_channels // 4, output_channels=output_channels // 4
            )
        self.l_block4 = LBlock(input_channels=output_channels // 4, output_channels=output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: tensor on the correct device, to move over the latent distribution

        Returns:

        """

        # Independent draws from Normal distribution
        # 独立正太取样
        z = self.distribution.sample(self.shape)
        # Batch is at end for some reason
        # reshape
        # torch.permute：按照指定的输入对张量进行维度的换位
        z = torch.permute(z, (3, 0, 1, 2)).type_as(x)

        # 3x3 Convolution
        z = self.conv_3x3(z)

        # 3 L Blocks to increase number of channels
        z = self.l_block1(z)
        z = self.l_block2(z)
        z = self.l_block3(z)
        # Spatial attention （空间 attention） module
        z = self.att_block(z)

        # L block to increase number of channel to 768
        z = self.l_block4(z)
        return z
