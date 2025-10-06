import torch
import torch.nn.functional as F
from torch import nn

from models.EGFM import EGFM
from models.CIFM import CIFM
from models.CNR import CGR
from models.DEAM import DEAM
# from thop import profile

class COD(nn.Module):
    def __init__(self, embed_dim=[32, 64, 128, 256, 512], depth=[3, 3, 3, 3, 3]):
        super(COD, self).__init__()

        self.deam = DEAM(embed_dim[1], embed_dim[4])

        self.cgr1 = CGR(embed_dim[0], embed_dim[1], kernel_size=3, stride=1, padding=1)
        self.cgr2 = CGR(embed_dim[1], embed_dim[2], kernel_size=3, stride=1, padding=1)
        self.cgr3 = CGR(embed_dim[2], embed_dim[3], kernel_size=3, stride=1, padding=1)
        self.cgr4 = CGR(embed_dim[3], embed_dim[4], kernel_size=3, stride=1, padding=1)

        self.egfm1 = EGFM(embed_dim[1])
        self.egfm2 = EGFM(embed_dim[2])
        self.egfm3 = EGFM(embed_dim[3])
        self.egfm4 = EGFM(embed_dim[4])

        self.conv_1 = CGR(embed_dim[1], embed_dim[1], kernel_size=1, stride=1, padding=0)
        self.conv_2 = CGR(embed_dim[2], embed_dim[2], kernel_size=1, stride=1, padding=0)
        self.conv_3 = CGR(embed_dim[3], embed_dim[3], kernel_size=1, stride=1, padding=0)
        self.conv_4 = CGR(embed_dim[4], embed_dim[4], kernel_size=1, stride=1, padding=0)

        self.cifm1 = CIFM(embed_dim[2], embed_dim[1])
        self.cifm2 = CIFM(embed_dim[3], embed_dim[2])
        self.cifm3 = CIFM(embed_dim[4], embed_dim[3])

        self.predictor1 = nn.Conv2d(embed_dim[1], 1, 1)
        self.predictor2 = nn.Conv2d(embed_dim[2], 1, 1)
        self.predictor3 = nn.Conv2d(embed_dim[3], 1, 1)

    def forward(self, x_list, output_size=None):
        x1 = x_list[0]
        x2 = x_list[1]
        x3 = x_list[2]
        x4 = x_list[3]

        if output_size is None:
            output_size = x1.shape[2:]  # 自动使用x1的(H, W)

        x1 = self.cgr1(x1)
        x2 = self.cgr2(x2)
        x3 = self.cgr3(x3)
        x4 = self.cgr4(x4)

        edge = self.deam(x1, x4)
        edge_att = torch.sigmoid(edge)

        x1_att = self.egfm1(x1, edge_att)
        x2_att = self.egfm2(x2, edge_att)
        x3_att = self.egfm3(x3, edge_att)
        x4_att = self.egfm4(x4, edge_att)

        x1r = self.conv_1(x1_att)
        x2r = self.conv_2(x2_att)
        x3r = self.conv_3(x3_att)
        x4r = self.conv_4(x4_att)

        x34 = self.cifm3(x3r, x4r)
        x234 = self.cifm2(x2r, x34)
        x1234 = self.cifm1(x1r, x234)

        pred3 = self.predictor3(x34)
        pred3 = F.interpolate(pred3, size=output_size, mode='bilinear', align_corners=False)
        pred2 = self.predictor2(x234)
        pred2 = F.interpolate(pred2, size=output_size, mode='bilinear', align_corners=False)
        pred1 = self.predictor1(x1234)
        pred1 = F.interpolate(pred1, size=output_size, mode='bilinear', align_corners=False)

        return pred1, pred2, pred3

# def test():
#     # 模拟编码器输出的多尺度特征图 (512x512输入)
#     x1 = torch.randn(1, 32, 512, 512).cuda()   # 原始尺寸
#     x2 = torch.randn(1, 64, 256, 256).cuda()   # 1/2下采样
#     x3 = torch.randn(1, 128, 128, 128).cuda()  # 1/4下采样
#     x4 = torch.randn(1, 256, 64, 64).cuda()    # 1/8下采样
    
#     x_list = [x1, x2, x3, x4]
    
#     # 创建模型
#     model = COD().cuda()
    
#     # 计算FLOPs和参数量
#     flops, params = profile(model, (x_list,))
#     print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
    
#     # 测试输出尺寸
#     preds = model(x_list)
#     print('输入: x1=%s, x2=%s, x3=%s, x4=%s' % (x1.shape, x2.shape, x3.shape, x4.shape))
#     print('输出: pred1=%s, pred2=%s, pred3=%s' % (preds[0].shape, preds[1].shape, preds[2].shape))
# if __name__ == "__main__":
#     test()