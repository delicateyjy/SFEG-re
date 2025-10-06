import torch
from torch import nn
import torch.nn.functional as F
from mmcls.SFEG_dev.models.SFEG.SFEG import SFEG
from models.COD import COD

class Decoder(nn.Module):

    def __init__(self, backbone, args=None):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.COD = COD()

    def forward(self, x):
        x_list = self.backbone(x)
        Mask1, Mask2, Mask3 = self.COD(x_list, output_size=x.shape[2:])
        return Mask1, Mask2, Mask3

class bic_iou(nn.Module):
    def __init__(self, args):
        super(bic_iou, self).__init__()
        self.args = args

    def forward(self, pred, true):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(true, kernel_size=31, stride=1, padding=15) - true)
        wbce = F.binary_cross_entropy_with_logits(pred, true, reduction='mean')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * true) * weit).sum(dim=(2, 3))
        union = ((pred + true) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (self.args.bce_weight * wbce + self.args.iou_weight * wiou).mean()

def build(args):
    # arg.device是str类型，将其转换为torch.device类型
    device = torch.device(args.device)
    args.device = device

    backbone = SFEG(arch='Base', 
                    in_chans=3, 
                    embed_dim=[32, 64, 128, 256, 512], 
                    depth=[3, 3, 3, 3, 3],
                    out_indices=(1, 2, 3, 4))

    model = Decoder(backbone, args)
    criterion = bic_iou(args)
    criterion.to(device)

    return model, criterion
