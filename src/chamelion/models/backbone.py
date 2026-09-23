#!/usr/bin/env python3
# @file      customminkunet.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import MinkowskiEngine as ME
import torch
import torch.nn as nn
from torch.optim import SGD


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        bn_momentum=0.1,
        dimension=-1,
    ):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
        )
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            dimension=dimension,
        )
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ConfidenceMinkUNet(nn.Module):
    BLOCK = BasicBlock
    PLANES = (8, 16, 32, 64, 64, 32, 16, 8)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    INIT_DIM = 8
    OUT_TENSOR_STRIDE = 1

    def m_space_n_time(self, m, n):
        return [m, m, m, n]

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels,
            self.inplanes,
            kernel_size=self.m_space_n_time(5, 1),
            dimension=D,
        )

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes,
            self.inplanes,
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes,
            self.inplanes,
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes,
            self.inplanes,
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes,
            self.inplanes,
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes,
            self.PLANES[4],
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2]
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes,
            self.PLANES[5],
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1]
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes,
            self.PLANES[6],
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0]
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes,
            self.PLANES[7],
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

        self.final_class = ME.MinkowskiConvolution(
            self.PLANES[7],
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D,
        )

        self.final_conf = ME.MinkowskiConvolution(
            self.PLANES[7],
            1,
            kernel_size=1,
            bias=True,
            dimension=D,
        )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks):
        downsample = None
        if self.inplanes != planes:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=1,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=1,
                dilation=1,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=1, dimension=self.D))

        return nn.Sequential(*layers)

    def forward(self, x):
        # * now we start encoder
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)  # N x 8

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)  # reduced N_a x 8

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)  # reduced N_b x 16

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)  # reduced N_c x 32

        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # * now we start decoder
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)

        return self.final_class(out), self.final_conf(out_b1p2)
