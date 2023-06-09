import torch
from torch import Tensor, nn


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleStrided(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int = 2) -> None:
        super().__init__()

        if not (2 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2

        self.branch1 = nn.Sequential(
            self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


class AugShuffleBlock(nn.Module):
    def __init__(self, inp: int, oup: int, r: float = 0.375) -> None:
        super().__init__()

        assert 0 < r <= 1, "r is a ratio"
        assert oup % 2 == 0, "Output channels needs to be divisible by 2"
        self.r = r
        self.bcf = branch_conv_features = int(oup * self.r)
        self.bbf = branch_bank_features = oup - branch_conv_features
        assert self.bcf % 2 == 0 and self.bbf % 2 == 0, f"Resulting features of each branch needs to be divisable by 2 ({self.bcf}, {self.bbf})"

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                branch_conv_features,
                branch_conv_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_conv_features),
            # M3: nn.ReLU(inplace=True),
            self.depthwise_conv(branch_conv_features, branch_conv_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(branch_conv_features)
        )

        # M2: banch Conv1x1
        self.branch3 = nn.Sequential(
            nn.Conv2d(oup//2, oup//2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup//2),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.split((self.bbf, self.bcf), dim=1)
        x2 = self.branch2(x2)

        # Crossover
        x11, x12 = x1.chunk(2, dim=1)
        x21, x22 = x2.chunk(2, dim=1)
        x1 = torch.cat((x21, x11), dim=1)
        x2 = torch.cat((x12, x22), dim=1)

        # Conv1x1, BN, ReLU
        x2 = self.branch3(x2)

        out = torch.cat((x1, x2), dim=1)
        out = channel_shuffle(out, 2)

        return out


class AugShuffleNet(nn.Module):
    def __init__(
        self,
        stages_repeats: list[int],
        stages_out_channels: list[int],
        input_channels: int = 3,
        r: float = 0.375
    ) -> None:
        super().__init__()

        if len(stages_out_channels) != len(stages_repeats)+1:
            raise ValueError(f"expected stages_out_channels as list of {len(stages_repeats)+1=} positive ints")
        self._stage_out_channels = stages_out_channels

        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = nn.ModuleList([])
        stage_names = [f"stage{i+2}" for i in range(len(stages_repeats))]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [ShuffleStrided(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(AugShuffleBlock(output_channels, output_channels))
            self.stages.append(nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        for stage in self.stages:
            x = stage(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class AugShuffleNet0_5x(AugShuffleNet):
    def __init__(self, input_channels=3):
        super().__init__([3, 7, 3], [24, 48, 96, 192], input_channels=input_channels)


class AugShuffleNet1_0x(AugShuffleNet):
    def __init__(self, input_channels=3):
        super().__init__([3, 7, 3], [24, 128, 256, 512], input_channels=input_channels)


class AugShuffleNet1_5x(AugShuffleNet):
    def __init__(self, input_channels=3):
        super().__init__([3, 7, 3], [24, 176, 352, 704], input_channels=input_channels)
