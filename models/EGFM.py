"""
EGFM（边缘引导融合模块）
有四个EGFM块，每个都接收DEAM输出的边界特征，以及分别接收四个SFA block输出的多尺度特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from models.CNR import Bott_CGR


class EGFM(nn.Module):
    def __init__(self, dim):
        super(EGFM, self).__init__()
        t = int(abs((log(dim, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv = Bott_CGR(dim, dim, dim // 8, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv(x)
        w = self.avg_pool(x)
        w = self.mlp(w.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        w = self.sigmoid(w)
        x = x * w

        return x
