"""
CIFM（跨尺度交互融合模块）
输入低层和高层特征，输出特征图
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from models.CNR import CBR

class CIFM(nn.Module):
    def __init__(self, hchannel, channel):
        super(CIFM, self).__init__()
        self.conv1_1 = CBR(hchannel + channel, channel, 1, padding=0)
        self.conv3_1 = CBR(channel // 4, channel // 4, 3, padding=1)
        self.depthwise5_1 = CBR(channel // 4, channel // 4, 3, dilation=2, padding=2)
        self.depthwise7_1 = CBR(channel // 4, channel // 4, 3, dilation=3, padding=3)
        self.depthwise9_1 = CBR(channel // 4, channel // 4, 3, dilation=4, padding=4)
        self.conv1_2 = CBR(channel, channel, 1, padding=0)
        self.conv3_2 = CBR(channel, channel, 3, padding=1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 16, channel, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)

        fusion_weight = nn.Parameter(torch.ones(2)).to(lf.device)
        fusion_weight = F.softmax(fusion_weight, dim=0)
        x = torch.cat((lf * fusion_weight[0], hf * fusion_weight[1]), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)

        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.depthwise5_1(xc[1] + x0 + xc[2])
        x2 = self.depthwise7_1(xc[2] + x1 + xc[3])
        x3 = self.depthwise9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_2(x + xx)

        ca = self.channel_attention(x)
        x = x * ca

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.spatial_attention(spatial)
        x = x * spatial + x

        return x