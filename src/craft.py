"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn.init as init
from torchutil import *

from basenet.vgg16_bn import vgg16_bn
from basenet.efficientnet_base import efficientnet_base

os.environ["CUDA_VISIBLE_DEVICES"]= "1"


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# # New Attention Block
# class AttentionBlock(nn.Module):
#     def __init__(self, in_ch, mid_ch, out_ch):
#         super(AttentionBlock, self).__init__()
#         self.conv_source = nn.Conv2d(in_ch, in_ch, kernel_size=1)
#         self.bn_source = nn.BatchNorm2d(in_ch)
#         self.conv_x = nn.Conv2d(mid_ch, mid_ch, kernel_size=1)
#         self.bn_x = nn.BatchNorm2d(mid_ch)

#         self.relu_concat = nn.ReLU()
#         self.conv_concat = nn.Conv2d(in_ch+mid_ch, in_ch+mid_ch, kernel_size=1)
#         self.bn_concat = nn.BatchNorm2d(in_ch+mid_ch)
#         self.sigmoid_concat = nn.Sigmoid()

#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
#             nn.BatchNorm2d(mid_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x, source = x

#         x = self.conv_x(x)
#         x = self.bn_x(x)
#         source = self.conv_source(source)
#         source = self.bn_source(source)

#         x = torch.cat([x, source], dim=1)

#         x = self.relu_concat(x)
#         x = self.conv_concat(x)
#         x = self.bn_concat(x)
#         x = self.sigmoid_concat(x)
        
#         x = self.double_conv(x)

#         return x


# New Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(AttentionBlock, self).__init__()
        self.attention1 = SCSEModule(in_ch+mid_ch)
        self.attention2 = SCSEModule(out_ch)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            # nn.GroupNorm(32,mid_ch),     # Num_groups = 32
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.GroupNorm(32,out_ch),    # Num_group = 32
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x, source = x

        x = torch.cat([x, source], dim=1)
        x = self.attention1(x)
        
        x = self.double_conv(x)
        x = self.attention2(x)

        return x


# Attention layer
class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch, ch//re, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch//re, ch, 1),
                                 nn.Sigmoid()
                                )
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1),
        # self.sSE = nn.Sequential(nn.Conv2d(ch, 1, 1),
                                 nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class CRAFT(nn.Module):
    def __init__(self, pretrained=True, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        # self.net = vgg16_bn(pretrained, freeze)
        # self.net.load_state_dict(copyStateDict(torch.load('vgg16_bn-6c64b313.pth')))
        # self.basenet = self.net
        self.basenet = vgg16_bn(pretrained, freeze)
        print('Backbone: VGG16')
        """ U network """
        self.upconv1 = AttentionBlock(1024, 512, 256)
        self.upconv2 = AttentionBlock(512, 256, 128)
        self.upconv3 = AttentionBlock(256, 128, 64)
        self.upconv4 = AttentionBlock(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        # y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1([sources[0], sources[1]])

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        # y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2([y, sources[2]])

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        # y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3([y, sources[3]])

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        # y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4([y, sources[4]])

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature


# # Config CRAFT with EfficientNet_B2
# class CRAFT(nn.Module):
#     def __init__(self, pretrained=True, freeze=False):
#         super(CRAFT, self).__init__()

#         """ Base network """
#         # self.net = vgg16_bn(pretrained, freeze)
#         # self.net.load_state_dict(copyStateDict(torch.load('vgg16_bn-6c64b313.pth')))
#         # self.basenet = self.net
#         self.basenet = efficientnet_base(pretrained, freeze)
#         """ U network """
#         self.upconv1 = AttentionBlock(448, 160, 256)
#         self.upconv2 = AttentionBlock(56, 256, 128)
#         self.upconv3 = AttentionBlock(32, 128, 64)
#         self.upconv4 = AttentionBlock(48, 64, 32)

#         num_class = 2
#         self.conv_cls = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
#             nn.Conv2d(16, num_class, kernel_size=1),
#         )

#         init_weights(self.upconv1.modules())
#         init_weights(self.upconv2.modules())
#         init_weights(self.upconv3.modules())
#         init_weights(self.upconv4.modules())
#         init_weights(self.conv_cls.modules())

#     def forward(self, x):
#         """ Base network """
#         sources = self.basenet(x)

#         """ U network """
#         y = F.interpolate(sources[0], size=sources[1].size()[2:], mode='bilinear', align_corners=False)
#         # y = torch.cat([y, sources[1]], dim=1)
#         y = self.upconv1([sources[1], y])

#         y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
#         # y = torch.cat([y, sources[2]], dim=1)
#         y = self.upconv2([y, sources[2]])

#         y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
#         # y = torch.cat([y, sources[3]], dim=1)
#         y = self.upconv3([y, sources[3]])

#         y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
#         # y = torch.cat([y, sources[4]], dim=1)
#         feature = self.upconv4([y, sources[4]])

#         y = self.conv_cls(feature)

#         return y.permute(0, 2, 3, 1), feature


if __name__ == '__main__':
    model = CRAFT(pretrained=False).cuda()
    output, _ = model(torch.randn(1, 3, 768, 768).cuda())
    print(output.shape)

