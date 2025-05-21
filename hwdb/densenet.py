"""自行复刻SideNet原文的DenseNet-44, 且带DSBN"""


import re
from collections import OrderedDict
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor


class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super().__init__()

        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.norm1_p = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.norm2_p = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs, modality) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        if modality != 'prn':
            bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        else:
            bottleneck_output = self.conv1(self.relu1(self.norm1_p(concated_features)))
        return bottleneck_output

    def forward(self, input, modality) -> Tensor: 
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features, modality)  # DSBN

        if modality != 'prn':  # DSBN
            new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        else:
            new_features = self.conv2(self.relu2(self.norm2_p(bottleneck_output)))

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()

        denselayers = list()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            denselayers.append(layer)
        self.denselayers = nn.ModuleList(denselayers)

    def forward(self, init_features, modality) -> Tensor:
        features = [init_features]
        for layer in self.denselayers:
            new_features = layer(features, modality)  # DSBN
            features.append(new_features)
        return torch.cat(features, 1)                 # 在channel维度进行拼接


class _Transition(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.norm_p = nn.BatchNorm2d(num_input_features)  
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x, modality):
        if modality != 'prn':
            x = self.norm(x)
        else:
            x = self.norm_p(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet44_DSBN(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (3, 4, 10, 3),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        memory_efficient: bool = False,
    ) -> None:

        super().__init__()

        # First convolution
        # self.conv0 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv0 = nn.Conv2d(1, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm0 = nn.BatchNorm2d(num_init_features)
        self.norm0_p = nn.BatchNorm2d(num_init_features)  # DSBN
        self.relu0 = nn.ReLU(inplace=True)
        # self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Each denseblock. _DenseBlock -> _DenseLayer
        num_features = num_init_features
        num_layers = block_config[0]
        self.denseblock1 =  _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
        )
        num_features = num_features + num_layers * growth_rate
        self.transition1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)  # transition layer
        num_features = num_features // 2  

        num_layers = block_config[1]
        self.denseblock2 =  _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
        )
        num_features = num_features + num_layers * growth_rate
        self.transition2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)  # transition layer
        num_features = num_features // 2

        num_layers = block_config[2]
        self.denseblock3 =  _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
        )
        num_features = num_features + num_layers * growth_rate
        self.transition3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)  # transition layer
        num_features = num_features // 2

        num_layers = block_config[3]
        self.denseblock4 =  _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
        )
        num_features = num_features + num_layers * growth_rate

        # Final batch norm
        self.norm5 = nn.BatchNorm2d(num_features)
        self.norm5_p = nn.BatchNorm2d(num_features)

        # # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, modality) -> Tensor:
        # features = self.features(x)
        x = self.conv0(x)
        if modality != 'prn':
            x = self.norm0(x)
        else:
            x = self.norm0_p(x)
        x = self.relu0(x)

        x = self.denseblock1(x, modality)
        x = self.transition1(x, modality)
        
        x = self.denseblock2(x, modality)
        x = self.transition2(x, modality)
        
        x = self.denseblock3(x, modality)
        x = self.transition3(x, modality)

        x = self.denseblock4(x, modality)
        
        if modality != 'prn':
            x = self.norm5(x)
        else:
            x = self.norm5_p(x)
        features = x

        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        # out = self.classifier(out)
        return out

