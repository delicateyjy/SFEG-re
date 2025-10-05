"""
DEAM（动态边缘感知模块）
输入SFA1和SFA4，高层和低层特征，输出边界特征，供给EGFM使用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from models.CNR import CGR, Bott_CGR

class DEAM(nn.Module):
    def __init__(self, lchannel, hchannel):
        super(DEAM, self).__init__()
        self.reduce1 = CGR(lchannel, lchannel, 1, padding=0)
        self.reduce4 = CGR(hchannel, hchannel // 2, 1, padding=0)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(lchannel + hchannel//2, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
            nn.Sigmoid()
        )
        self.balance_factor = nn.Parameter(torch.ones(1) * 0.5)
        self.block = nn.Sequential(
            Bott_CGR(hchannel // 2 + lchannel, hchannel // 2, hchannel // 16, kernel_size=3, stride=1, padding=1),
            Bott_CGR(hchannel // 2, hchannel // 2, hchannel // 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(hchannel // 2, 1, 1))

    def forward(self, x1, x4):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)

        concat_features = torch.cat([x1, x4], dim=1)
        weights = self.channel_attention(concat_features)
        w1, w4 = weights[:, 0:1, :, :], weights[:, 1:2, :, :]

        weight_sum = w1 + w4
        scale = 2.0 / (weight_sum + 1e-8)

        x1 = x1 * (w1 * scale).expand_as(x1) * self.balance_factor
        x4 = x4 * (w4 * scale).expand_as(x4) * (1 - self.balance_factor)

        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out
