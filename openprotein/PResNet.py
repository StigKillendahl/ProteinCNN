# This file is part of the ProteinCNN project.
#
# @author Stig Killendahl & Kevin Jon Jensen
#
# Based on the OpenProtein framework, please see the LICENSE file in the root directory.

import torch.nn as nn
import torch
import math
import openprotein
from util import *
from models import soft_to_angle


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def convkx1(in_planes, out_planes, kernel, padding, stride = 1):
    """
        kx1 convolution with padding p
    """
    return nn.Conv1d(in_planes, out_planes, kernel, padding=padding, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, kernel=(5,11), padding=(2,5), stride=1, downsample=None, droprate=0.0):

        super(Bottleneck, self).__init__()
        k1,k2 = kernel[0], kernel[1]
        p1,p2 = padding[0], padding[1]
        self.conv1 = convkx1(inplanes, planes, k1, p1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = convkx1(planes, planes, k2,p2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = convkx1(planes, planes*4, k1, p1)
        self.bn3 = nn.BatchNorm1d(planes*4)

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel=11, padding=5, stride=1, droprate=0.0, downsample=None):
        super(BasicBlock, self).__init__()
        self.droprate=droprate
        self.dropout = nn.Dropout(p=self.droprate)
        self.conv1 = convkx1(inplanes, planes, kernel, padding, stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.conv2 = convkx1(planes, planes, kernel, padding)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.droprate > 0.0:
            out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = torch.add(out, residual)

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, input_channels=21, droprate=[0.0,0.0,0.0,0.0,0.0], out_channels=500, kernel=5, padding=2, stride=1, use_gpu=False):
        self.inplanes = 64
        self.depth = input_channels
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=11, stride=1, padding=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.soft = nn.LogSoftmax(2)
        self.softmax_to_angle = soft_to_angle(out_channels)
        self.input_droprate = droprate[0]
        self.input_dropout = nn.Dropout(p=droprate[0])
        self.layer1 = self._make_layer(block, 64, layers[0], kernel=kernel, padding=padding, droprate=droprate[1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride, kernel=kernel, padding=padding, droprate=droprate[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride, kernel=kernel, padding=padding, droprate=droprate[3])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride, kernel=kernel, padding=padding, droprate=droprate[4])

        self.convOut = nn.Conv1d(self.inplanes, out_channels, kernel_size=5, padding=2)
        self.use_gpu = use_gpu

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0]  * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel=11, padding=5, droprate=0.0):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        
        layers = []

        layers.append(block(self.inplanes, planes, kernel=kernel, padding=padding, stride=stride, downsample=downsample, droprate=droprate))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel=kernel, padding=padding, droprate=droprate))

        return nn.Sequential(*layers)


    def _get_network_emissions(self, amino_acids):
        data, batch_sizes = amino_acids
        x = data.transpose(0,1).transpose(1,2)

        # print(x.size())
        if self.input_droprate > 0.0:
            x = self.input_dropout(x)
        x = self.conv1(x)
        # print(x.size())
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        # print(x.size())
        x = self.layer2(x)
        # print(x.size())
        x = self.layer3(x)
        # print(x.size())
        x = self.layer4(x)
        # print(x.size())

        x = self.soft(self.convOut(x))

        x = x.transpose(1,2)
        p = torch.exp(x)

        # output_angles = self.softmax_to_angle(p).transpose(0,1) # max size, minibatch size, 3 (angels)
        # backbone_atoms_padded, batch_sizes_backbone = get_backbone_positions_from_angular_prediction(output_angles, batch_sizes, self.use_gpu)
        return p, batch_sizes


def resnet18(**kwargs):
    """ Construct a ResNet-18 model.

        Args:
            kwargs:
                input_channels: number of input channels.
                out_channels: final out channels
                kernel: kernel for each block
                padding: padding for each basicblock
                stride: stride for changing output size (W)
                use_gpu: Bool

    """
    model = ResNet(BasicBlock, [2,2,2,2], **kwargs)

    return model

def resnet34(**kwargs):
    """ Construct a ResNet-34 model.
        Args:
            kwargs:
                input_channels: number of input channels.
                out_channels: final out channels
                kernel: kernel for each block
                padding: padding for each basicblock
                stride: stride for changing output size (W)
                use_gpu: Bool
    """

    model = ResNet(BasicBlock, [3,4,6,3], **kwargs)

    return model


def resnet50(**kwargs):
    """ Construct a ResNet-50 model.
        Args:
            kwargs:
                input_channels: number of input channels.
                out_channels: final out channels
                kernel: kernel for each block (tuple)
                padding: padding for each basicblock (tuple)
                stride: stride for changing output size (W)
                use_gpu: Bool
    """
    model = ResNet(Bottleneck, [3,4,6,3], **kwargs)

    return model

def resnet101(**kwargs):
    """ Construct a ResNet-101 model.
        Args:
            kwargs:
                input_channels: number of input channels.
                out_channels: final out channels
                kernel: kernel for each block (tuple)
                padding: padding for each basicblock (tuple)
                stride: stride for changing output size (W)
                use_gpu: Bool
    """

    model = ResNet(Bottleneck, [3,4,23,3], **kwargs)
    return model

def resnet152(**kwargs):
    """ Construct a ResNet-152 model.
        Args:
            kwargs:
                input_channels: number of input channels.
                out_channels: final out channels
                kernel: kernel for each block (tuple)
                padding: padding for each basicblock (tuple)
                stride: stride for changing output size (W)
                use_gpu: Bool
    """
    model = ResNet(Bottleneck, [3,8,36,3], **kwargs)
    return model


name_dict = {'resnet18':resnet18, 'resnet34':resnet34, 'resnet50':resnet50, 'resnet101':resnet101, 'resnet152':resnet152}


