import torch
from torch.nn.modules.pixelshuffle import PixelUnshuffle
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.functional as F
from dgmr.common import DBlock
from huggingface_hub import PyTorchModelHubMixin


# 空间与时间辨别器的整合应用，计算二者预测输出之和
class Discriminator(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_channels: int = 12,
        num_spatial_frames: int = 8,
        conv_type: str = "standard",
        **kwargs
    ):
        super().__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        num_spatial_frames = self.config["num_spatial_frames"]
        conv_type = self.config["conv_type"]

        # 空间辨别器
        self.spatial_discriminator = SpatialDiscriminator(
            input_channels=input_channels, num_timesteps=num_spatial_frames, conv_type=conv_type
        )

        # 时间辨别器
        self.temporal_discriminator = TemporalDiscriminator(
            input_channels=input_channels, conv_type=conv_type
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_loss = self.spatial_discriminator(x)
        temporal_loss = self.temporal_discriminator(x)

        # 2*1
        # 两个生成器预测输出之和
        return torch.cat([spatial_loss, temporal_loss], dim=1)


# 时间辨别器
# DS--S2D--d1--d2--（ds--d_last--bn--fc）for Timesteps
class TemporalDiscriminator(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, input_channels: int = 12, num_layers: int = 3, conv_type: str = "standard", **kwargs
    ):
        """
        Temporal Discriminator from the Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        Args:
            input_channels: Number of channels per timestep
            crop_size: Size of the crop, in the paper half the width of the input images
                        图片裁剪大小，宽高是输入图片的一半
            num_layers: Number of intermediate DBlock layers to use
                        中间要使用的 DBlock 层数
            conv_type: Type of 2d convolutions to use, see satflow/models/utils.py for options
        """
        super().__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        num_layers = self.config["num_layers"]
        conv_type = self.config["conv_type"]
        # 定义下采样为 3D 平均池化样本
        self.downsample = torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # 重新改变tensor形状，(∗,C,H×r,W×r) to a tensor of shape (*, C * r^2, H, W) where r is the downscale_factor.
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        internal_chn = 48

        self.d1 = DBlock(
            input_channels=4 * input_channels,
            output_channels=internal_chn * input_channels,
            conv_type="3d",
            first_relu=False,
        )

        self.d2 = DBlock(
            input_channels=internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            conv_type="3d",
        )

        self.intermediate_dblocks = torch.nn.ModuleList()
        # 根据之后要使用的 DBlock 层数进行填充
        for _ in range(num_layers):
            internal_chn *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    input_channels=internal_chn * input_channels,
                    output_channels=2 * internal_chn * input_channels,
                    conv_type=conv_type,
                )
            )

        self.d_last = DBlock(
            input_channels=2 * internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )

        # 线性后谱归一化
        self.fc = spectral_norm(torch.nn.Linear(2 * internal_chn * input_channels, 1))
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(2 * internal_chn * input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)

        x = self.space2depth(x)
        # Have to move time and channels
        # 改变 时间步和通道的位置
        x = torch.permute(x, dims=(0, 2, 1, 3, 4))
        # 2 residual 3D blocks to halve resolution of image, double number of channels and reduce number of time steps
        # 两个 3D 残差块 使得图像分辨率减半，通道数加倍，减少时间步数
        x = self.d1(x)
        x = self.d2(x)
        # Convert back to T x C x H x W
        # 还原
        x = torch.permute(x, dims=(0, 2, 1, 3, 4))
        # Per Timestep part now, same as spatial discriminator
        # 遍历每一个时间步
        representations = []
        for idx in range(x.size(1)):
            # Intermediate DBlocks
            # Three residual D Blocks to halve the resolution of the image and double the number of channels.
            rep = x[:, idx, :, :, :]
            # 使用中间层的残差块进行处理,从图像来看是三个
            for d in self.intermediate_dblocks:
                rep = d(rep)
            # One more D Block without downsampling or increase number of channels
            rep = self.d_last(rep)

            rep = torch.sum(F.relu(rep), dim=[2, 3])
            rep = self.bn(rep)
            rep = self.fc(rep)

            representations.append(rep)
        # The representations are summed together before the ReLU
        x = torch.stack(representations, dim=1)
        # Should be [Batch, N, 1]
        x = torch.sum(x, keepdim=True, dim=1)
        return x


# 空间辨别器
class SpatialDiscriminator(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_channels: int = 12,
        num_timesteps: int = 8,
        num_layers: int = 4,
        conv_type: str = "standard",
        **kwargs
    ):
        """
        Spatial discriminator from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        Args:
            input_channels: Number of input channels per timestep
            num_timesteps: Number of timesteps to use, in the paper 8/18 timesteps were chosen
            num_layers: Number of intermediate DBlock layers to use
            conv_type: Type of 2d convolutions to use, see satflow/models/utils.py for options
        """
        super().__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        num_timesteps = self.config["num_timesteps"]
        num_layers = self.config["num_layers"]
        conv_type = self.config["conv_type"]
        # Randomly, uniformly, select 8 timesteps to do this on from the input
        self.num_timesteps = num_timesteps

        # First step is mean pooling 2x2 to reduce input by half
        self.mean_pool = torch.nn.AvgPool2d(2)
        self.space2depth = PixelUnshuffle(downscale_factor=2)

        internal_chn = 24
        self.d1 = DBlock(
            input_channels=4 * input_channels,
            output_channels=2 * internal_chn * input_channels,
            first_relu=False,
            conv_type=conv_type,
        )
        self.intermediate_dblocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            internal_chn *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    input_channels=internal_chn * input_channels,
                    output_channels=2 * internal_chn * input_channels,
                    conv_type=conv_type,
                )
            )
        self.d6 = DBlock(
            input_channels=2 * internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )

        # Spectrally normalized linear layer for binary classification
        self.fc = spectral_norm(torch.nn.Linear(2 * internal_chn * input_channels, 1))
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(2 * internal_chn * input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x should be the chosen 8 or so
        idxs = torch.randint(low=0, high=x.size()[1], size=(self.num_timesteps,))
        representations = []
        for idx in idxs:
            rep = self.mean_pool(x[:, idx, :, :, :])  # 128x128
            rep = self.space2depth(rep)  # 64x64x4
            rep = self.d1(rep)  # 32x32
            # Intermediate DBlocks
            for d in self.intermediate_dblocks:
                rep = d(rep)
            rep = self.d6(rep)  # 2x2
            rep = torch.sum(F.relu(rep), dim=[2, 3])
            rep = self.bn(rep)
            # fc 后是一个值即图片为真的概率
            rep = self.fc(rep)
            """
            Pseudocode from DeepMind
            # Sum-pool the representations and feed to spectrally normalized lin. layer.
            y = tf.reduce_sum(tf.nn.relu(y), axis=[1, 2])
            y = layers.BatchNorm(calc_sigma=False)(y)
            output_layer = layers.Linear(output_size=1)
            output = output_layer(y)

            # Take the sum across the t samples. Note: we apply the ReLU to
            # (1 - score_real) and (1 + score_generated) in the loss.
            output = tf.reshape(output, [b, n, 1])
            output = tf.reduce_sum(output, keepdims=True, axis=1)
            return output
            """
            representations.append(rep)

        # The representations are summed together before the ReLU
        x = torch.stack(representations, dim=1)
        # Should be [Batch, N, 1]
        # 所有预测结果的和
        x = torch.sum(x, keepdim=True, dim=1)
        return x
