import torch
import torch.nn as nn

import numpy as np


class DSBNResBlock(nn.Module):
    def __init__(self, nFin, nFout):
        super(DSBNResBlock, self).__init__()

        # self.conv_block = nn.Sequential()
        # self.conv_block.add_module('BNorm1', nn.BatchNorm2d(nFin))
        # self.conv_block.add_module('LRelu1', nn.LeakyReLU(0.2))
        # self.conv_block.add_module('ConvL1', nn.Conv2d(nFin,  nFout, kernel_size=3, padding=1, bias=False))
        # self.conv_block.add_module('BNorm2', nn.BatchNorm2d(nFout))
        # self.conv_block.add_module('LRelu2', nn.LeakyReLU(0.2))
        # self.conv_block.add_module('ConvL2', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))
        # self.conv_block.add_module('BNorm3', nn.BatchNorm2d(nFout))
        # self.conv_block.add_module('LRelu3', nn.LeakyReLU(0.2))
        # self.conv_block.add_module('ConvL3', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))
        self.bn1_bns_0 = nn.BatchNorm2d(nFin)
        self.bn1_bns_1 = nn.BatchNorm2d(nFin)
        
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(nFin,  nFout, kernel_size=3, padding=1, bias=False)
        
        self.bn2_bns_0 = nn.BatchNorm2d(nFout)
        self.bn2_bns_1 = nn.BatchNorm2d(nFout)
        
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False)
        
        self.bn3_bns_0 = nn.BatchNorm2d(nFout)
        self.bn3_bns_1 = nn.BatchNorm2d(nFout)
        
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False)

        self.skip_layer = nn.Conv2d(nFin, nFout, kernel_size=1, stride=1)

    def forward(self, x, modality):
        x0 = x
        if modality != 'prn': 
            x = self.bn1_bns_0(x)
        else:
            x = self.bn1_bns_1(x)
        x = self.lrelu1(x)
        x = self.conv1(x)
        
        if modality != 'prn': 
            x = self.bn2_bns_0(x)
        else:
            x = self.bn2_bns_1(x)
        
        x = self.lrelu2(x)
        x = self.conv2(x)
        
        if modality != 'prn': 
            x = self.bn3_bns_0(x)
        else:
            x = self.bn3_bns_1(x)
        
        x = self.lrelu3(x)
        x = self.conv3(x)

        x = self.skip_layer(x0) + x
        
        # return self.skip_layer(x) + self.conv_block(x)
        return x, modality


class ResNetLike(nn.Module):
    def __init__(self, opt):
        super(ResNetLike, self).__init__()

        self.in_planes = 1
        self.out_planes = [64, 128, 256, 512]
        self.num_stages = 4

        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]

        assert(type(self.out_planes)==list)
        assert(len(self.out_planes)==self.num_stages)
        num_planes = [self.out_planes[0],] + self.out_planes

        self.conv0 = nn.Conv2d(self.in_planes, num_planes[0], kernel_size=3, padding=1)
        self.dsbnresblock0 = DSBNResBlock(num_planes[0], num_planes[1])
        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dsbnresblock1 = DSBNResBlock(num_planes[1], num_planes[1+1])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dsbnresblock2 = DSBNResBlock(num_planes[2], num_planes[2+1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dsbnresblock3 = DSBNResBlock(num_planes[3], num_planes[3+1])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.lrelu1 = nn.LeakyReLU(0.2, True)

        # 降维
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.feaDim = opt.feaDim
        if self.feaDim < 512:
            self.fc = nn.Linear(512, self.feaDim)

    def forward(self, x, modality='hw'):    
        x = self.conv0(x)
        x, _ = self.dsbnresblock0(x, modality)
        x = self.maxpool0(x)
        x, _ = self.dsbnresblock1(x, modality)
        x = self.maxpool1(x)
        x, _ = self.dsbnresblock2(x, modality)
        x = self.maxpool2(x)
        x, _ = self.dsbnresblock3(x, modality)
        x = self.maxpool3(x)
        x = self.lrelu1(x)

        # 降维
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        if self.feaDim < 512:
            x = self.fc(x)
        return x


class ResNetLikeV2(nn.Module):
    """网络结构加深, 22层=1+(1+2+2+2)*3, 提点冲击Sota, 难以维持"""
    def __init__(self, opt):
        super(ResNetLikeV2, self).__init__()

        self.in_planes = 1
        self.out_planes = [64, 128, 256, 512]
        self.num_stages = 4

        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]

        assert(type(self.out_planes)==list)
        assert(len(self.out_planes)==self.num_stages)
        num_planes = [self.out_planes[0],] + self.out_planes

        self.conv0 = nn.Conv2d(self.in_planes, num_planes[0], kernel_size=3, padding=1)
        
        self.dsbnresblock0 = DSBNResBlock(num_planes[0], num_planes[1])
        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dsbnresblock1 = DSBNResBlock(num_planes[1], num_planes[1])
        self.dsbnresblock1_1 = DSBNResBlock(num_planes[1], num_planes[2])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dsbnresblock2 = DSBNResBlock(num_planes[2], num_planes[2])
        self.dsbnresblock2_1 = DSBNResBlock(num_planes[2], num_planes[3])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dsbnresblock3 = DSBNResBlock(num_planes[3], num_planes[3])
        self.dsbnresblock3_1 = DSBNResBlock(num_planes[3], num_planes[4])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.lrelu1 = nn.LeakyReLU(0.2, True)

        # 降维
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.feaDim = opt.feaDim
        if self.feaDim < 512:
            self.fc = nn.Linear(512, self.feaDim)

    def forward(self, x, modality='hw'):
        x = self.conv0(x)
        x, _ = self.dsbnresblock0(x, modality)
        x = self.maxpool0(x)
        x, _ = self.dsbnresblock1(x, modality)
        x, _ = self.dsbnresblock1_1(x, modality)
        x = self.maxpool1(x)
        x, _ = self.dsbnresblock2(x, modality)
        x, _ = self.dsbnresblock2_1(x, modality)
        x = self.maxpool2(x)
        x, _ = self.dsbnresblock3(x, modality)
        x, _ = self.dsbnresblock3_1(x, modality)
        x = self.maxpool3(x)
        
        x = self.lrelu1(x)

        # 降维
        x = self.maxpool(x)   # 空间
        x = x.view(x.size(0), -1)
        if self.feaDim < 512:
            x = self.fc(x)
        return x


class ResNetLikeV3(nn.Module):
    """网络结构加深, 19层=1+(1+1+2+2)*3, 提点冲击Sota"""
    def __init__(self, opt):
        super(ResNetLikeV3, self).__init__()

        self.in_planes = 1
        self.out_planes = [64, 128, 256, 512]
        self.num_stages = 4

        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]

        assert(type(self.out_planes)==list)
        assert(len(self.out_planes)==self.num_stages)
        num_planes = [self.out_planes[0],] + self.out_planes

        self.conv0 = nn.Conv2d(self.in_planes, num_planes[0], kernel_size=3, padding=1)
        
        self.dsbnresblock0 = DSBNResBlock(num_planes[0], num_planes[1])
        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dsbnresblock1 = DSBNResBlock(num_planes[1], num_planes[2])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dsbnresblock2 = DSBNResBlock(num_planes[2], num_planes[2])
        self.dsbnresblock2_1 = DSBNResBlock(num_planes[2], num_planes[3])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dsbnresblock3 = DSBNResBlock(num_planes[3], num_planes[3])
        self.dsbnresblock3_1 = DSBNResBlock(num_planes[3], num_planes[4])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.lrelu1 = nn.LeakyReLU(0.2, True)

        # 降维
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.feaDim = opt.feaDim
        if self.feaDim < 512:
            self.fc = nn.Linear(512, self.feaDim)

    def forward(self, x, modality='hw'):
        x = self.conv0(x)
        x, _ = self.dsbnresblock0(x, modality)
        x = self.maxpool0(x)
        x, _ = self.dsbnresblock1(x, modality)
        x = self.maxpool1(x)
        x, _ = self.dsbnresblock2(x, modality)
        x, _ = self.dsbnresblock2_1(x, modality)
        x = self.maxpool2(x)
        x, _ = self.dsbnresblock3(x, modality)
        x, _ = self.dsbnresblock3_1(x, modality)
        x = self.maxpool3(x)
        
        x = self.lrelu1(x)

        # 降维
        x = self.maxpool(x)   # 空间
        x = x.view(x.size(0), -1)
        if self.feaDim < 512:
            x = self.fc(x)
        return x


class ResNetLikeV4(nn.Module):
    """网络结构加深, 16层=1+(1+1+1+2)*3, 提点冲击Sota, 难以维持"""
    def __init__(self, opt):
        super(ResNetLikeV4, self).__init__()

        self.in_planes = 1
        self.out_planes = [64, 128, 256, 512]
        self.num_stages = 4

        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]

        assert(type(self.out_planes)==list)
        assert(len(self.out_planes)==self.num_stages)
        num_planes = [self.out_planes[0],] + self.out_planes

        self.conv0 = nn.Conv2d(self.in_planes, num_planes[0], kernel_size=3, padding=1)
        
        self.dsbnresblock0 = DSBNResBlock(num_planes[0], num_planes[1])
        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dsbnresblock1 = DSBNResBlock(num_planes[1], num_planes[2])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dsbnresblock2 = DSBNResBlock(num_planes[2], num_planes[3])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dsbnresblock3 = DSBNResBlock(num_planes[3], num_planes[3])
        self.dsbnresblock3_1 = DSBNResBlock(num_planes[3], num_planes[4])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.lrelu1 = nn.LeakyReLU(0.2, True)

        # 降维
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.feaDim = opt.feaDim
        if self.feaDim < 512:
            self.fc = nn.Linear(512, self.feaDim)

    def forward(self, x, modality='hw'):
        x = self.conv0(x)
        x, _ = self.dsbnresblock0(x, modality)
        x = self.maxpool0(x)
        x, _ = self.dsbnresblock1(x, modality)
        x = self.maxpool1(x)
        x, _ = self.dsbnresblock2(x, modality)
        x = self.maxpool2(x)
        x, _ = self.dsbnresblock3(x, modality)
        x, _ = self.dsbnresblock3_1(x, modality)
        x = self.maxpool3(x)
        
        x = self.lrelu1(x)

        # 降维
        x = self.maxpool(x)   # 空间
        x = x.view(x.size(0), -1)
        if self.feaDim < 512:
            x = self.fc(x)
        return x



