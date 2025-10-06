import torch.nn as nn
from models.SFPM import Spatial_Fourier_Parallel_Mixer
from models.PFGM import Prior_Feed_Gate_Mixer

class SFA_Block(nn.Module):
    """空间特征聚合基础块"""
    def __init__(self, dim):
        super(SFA_Block, self).__init__()
        self.dim = dim
        self.norm1 = nn.GroupNorm(32, dim)
        self.norm2 = nn.GroupNorm(32, dim)
        self.mixer = Spatial_Fourier_Parallel_Mixer(dim)
        self.ffn = Prior_Feed_Gate_Mixer(dim)

    def forward(self, mix_input):
        x, mask = mix_input
        
        # 第一个残差分支
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x + copy

        # 第二个残差分支
        copy = x
        x = self.norm2(x)
        x, mask = self.ffn(x, mask)
        x = x + copy

        return x, mask


class SFA_Stage(nn.Module):
    """SFA_Block的多层堆叠"""
    def __init__(self, depth, in_channels):
        super(SFA_Stage, self).__init__()
        self.blocks = nn.Sequential(
            *[SFA_Block(dim=in_channels) for _ in range(depth)]
        )

    def forward(self, mix_input):
        return self.blocks(mix_input)

# class SFA_Backbone(nn.Module):
#     def __init__(self, 
#                  in_chans=3, 
#                  patch_size=1, 
#                  embed_dim=[32, 64, 128, 256, 512], 
#                  depth=[3, 3, 3, 3, 3], 
#                  embed_kernel_size=3):
#         super(SFA_Backbone, self).__init__()
#         self.patch_size = patch_size
#         self.patch_embed = PatchEmbed(in_chans=in_chans, patch_size=patch_size,
#                                       embed_dim=embed_dim[0], kernel_size=embed_kernel_size)
        
#         self.layer1 = SFA_Stage(depth=depth[0], in_channels=embed_dim[0])
#         self.downsample1 = DownSample(input_dim=embed_dim[0])
#         self.down_rcp1 = DownSample_CP()

#         self.layer2 = SFA_Stage(depth=depth[1], in_channels=embed_dim[1])
#         self.downsample2 = DownSample(input_dim=embed_dim[1])
#         self.down_rcp2 = DownSample_CP()

#         self.layer3 = SFA_Stage(depth=depth[2], in_channels=embed_dim[2])
#         self.downsample3 = DownSample(input_dim=embed_dim[2])
#         self.down_rcp3 = DownSample_CP()

#         self.layer4 = SFA_Stage(depth=depth[3], in_channels=embed_dim[3])
#         self.downsample4 = DownSample(input_dim=embed_dim[3])
#         self.down_rcp4 = DownSample_CP()

#     def forward(self, x):
#         layer_output = []
#         layer_after_downsample_output = []

#         mask = get_gradient_prior(x)
#         x = self.patch_embed(x)

#         x, mask = self.layer1((x, mask))
#         layer_output.append(x)
#         x = self.downsample1(x)
#         layer_after_downsample_output.append(x)
#         mask = self.down_rcp1(mask)

#         x, mask = self.layer2((x, mask))
#         layer_output.append(x)
#         x = self.downsample2(x)
#         layer_after_downsample_output.append(x)
#         mask = self.down_rcp2(mask)

#         x, mask = self.layer3((x, mask))
#         layer_output.append(x)
#         x = self.downsample3(x)
#         layer_after_downsample_output.append(x)
#         mask = self.down_rcp3(mask)

#         x, mask = self.layer4((x, mask))
#         layer_output.append(x)
#         x = self.downsample4(x)
#         layer_after_downsample_output.append(x)
#         mask = self.down_rcp4(mask)

#         return layer_output