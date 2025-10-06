import torch
import torch.nn as nn
from mmcls.models.builder import BACKBONES
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.utils import to_2tuple

# 导入外部模块
from models.SFA import SFA_Stage
from models.patch_embed import PatchEmbed
from models.downsample import DownSample, DownSample_CP
from models.gradient_utils import get_gradient_prior

@BACKBONES.register_module()
class SFEG(BaseBackbone):
    """
    Spatial Feature Aggregation Backbone
    用于分割/边缘检测的编码器架构
    """
    # 各种配置选项
    arch_zoo = {
        'Base': {
            'embed_dim': [32, 64, 128, 256, 512],
            'depth': [3, 3, 3, 3, 3]
        },
        'Large': {
            'embed_dim': [64, 128, 256, 512, 1024],
            'depth': [4, 4, 6, 4, 4]
        }
    }
    
    def __init__(self,
                 in_chans=3,
                 arch=None,
                 patch_size=1,
                 embed_dim=[32, 64, 128, 256, 512],
                 depth=[3, 3, 3, 3, 3],
                 embed_kernel_size=3,
                 out_indices=(1, 2, 3, 4),  # ← 新增：支持多尺度输出
                 init_cfg=None,
                 **kwargs):
        super(SFEG, self).__init__(init_cfg)
        
        # 处理arch_zoo配置
        if arch is not None:
            assert arch in self.arch_zoo.keys()
            embed_dim = self.arch_zoo[arch]['embed_dim']
            depth = self.arch_zoo[arch]['depth']
        
        self.patch_size = patch_size
        self.out_indices = out_indices
        
        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            patch_size=patch_size,
            embed_dim=embed_dim[0],
            kernel_size=embed_kernel_size
        )
        
        # 2. 构建各个Stage
        self.layer1 = SFA_Stage(depth=depth[0], in_channels=embed_dim[0])
        self.downsample1 = DownSample(input_dim=embed_dim[0])
        self.down_rcp1 = DownSample_CP()
        
        self.layer2 = SFA_Stage(depth=depth[1], in_channels=embed_dim[1])
        self.downsample2 = DownSample(input_dim=embed_dim[1])
        self.down_rcp2 = DownSample_CP()
        
        self.layer3 = SFA_Stage(depth=depth[2], in_channels=embed_dim[2])
        self.downsample3 = DownSample(input_dim=embed_dim[2])
        self.down_rcp3 = DownSample_CP()
        
        self.layer4 = SFA_Stage(depth=depth[3], in_channels=embed_dim[3])
        self.downsample4 = DownSample(input_dim=embed_dim[3])
        self.down_rcp4 = DownSample_CP()
    
    # def init_weights(self):
    #     """权重初始化"""
    #     super(SFEG, self).init_weights()
        
    #     if not (isinstance(self.init_cfg, dict) and 
    #             self.init_cfg.get('type') == 'Pretrained'):
    #         # 自定义初始化
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
    #                 if m.bias is not None:
    #                     nn.init.constant_(m.bias, 0)
    #             elif isinstance(m, nn.GroupNorm):
    #                 nn.init.constant_(m.weight, 1)
    #                 nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        输入:
            x: (B, 3, H, W)（H，W设为512）
        输出:
            tuple[Tensor]: 多尺度特征
                - layer0: (B, 32, H, W)
                - layer1: (B, 64, H/2, W/2)
                - layer2: (B, 128, H/4, W/4)
                - layer3: (B, 256, H/8, W/8)
        """
        outs = []
        
        # 提取梯度先验
        mask = get_gradient_prior(x)
        
        # Patch Embedding
        x = self.patch_embed(x)
        
        # Stage 1
        x, mask = self.layer1((x, mask))
        if 1 in self.out_indices:
            outs.append(x)
        x = self.downsample1(x)
        mask = self.down_rcp1(mask)
        
        # Stage 2
        x, mask = self.layer2((x, mask))
        if 2 in self.out_indices:
            outs.append(x)
        x = self.downsample2(x)
        mask = self.down_rcp2(mask)
        
        # Stage 3
        x, mask = self.layer3((x, mask))
        if 3 in self.out_indices:
            outs.append(x)
        x = self.downsample3(x)
        mask = self.down_rcp3(mask)
        
        # Stage 4
        x, mask = self.layer4((x, mask))
        if 4 in self.out_indices:
            outs.append(x)
        x = self.downsample4(x)
        mask = self.down_rcp4(mask)

        return outs
