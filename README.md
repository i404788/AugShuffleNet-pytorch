# AugShuffleNet: Communicate More, Compute Less

AugShuffle1_0x has slightly different channels than the paper due to constraints in of the default `r`.

See: <https://arxiv.org/abs/2203.06589>

## Usage
```
import torch
from augshufflenet_pytorch import AugShuffleNet0_5x, AugShuffleNet1_0x, AugShuffleNet1_5x, AugShuffleNet


model = AugShuffleNet0_5x(input_channels=3)
x = model(torch.randn(1, 3, 64, 64)) # [1, 192]

# Equivalent to 0_5x
model = AugShuffleNet(stages_repeats=[3, 7, 3], stages_out_channels=[24, 48, 96, 192], input_channels=3, r=0.375)
x = model(torch.randn(1, 3, 64, 64)) # [1, 192]
```

> NOTE: each of the int(out_channels * r) & out_channels putneeds to be divisible by 2

## Citation
```
@misc{ye_augshufflenet_2022,
	title = {{AugShuffleNet}: {Communicate} {More}, {Compute} {Less}},
	shorttitle = {{AugShuffleNet}},
	url = {http://arxiv.org/abs/2203.06589},
	doi = {10.48550/arXiv.2203.06589},
	abstract = {As a remarkable compact model, ShuffleNetV2 offers a good example to design efficient ConvNets but its limit is rarely noticed. In this paper, we rethink the design pattern of ShuffleNetV2 and find that the channel-wise redundancy problem still constrains the efficiency improvement of Shuffle block in the wider ShuffleNetV2. To resolve this issue, we propose another augmented variant of shuffle block in the form of bottleneck-like structure and more implicit short connections. To verify the effectiveness of this building block, we further build a more powerful and efficient model family, termed as AugShuffleNets. Evaluated on the CIFAR-10 and CIFAR-100 datasets, AugShuffleNet consistently outperforms ShuffleNetV2 in terms of accuracy with less computational cost and fewer parameter count.},
	urldate = {2023-06-09},
	publisher = {arXiv},
	author = {Ye, Longqing},
	month = aug,
	year = {2022},
	note = {arXiv:2203.06589 [cs]},
	keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning},
}
```