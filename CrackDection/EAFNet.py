import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.models as models
from torch.nn import init
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import os
from thop import profile

class TinyReLU(nn.Module):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        out = torch.where(
            x >= 0,
            x - torch.exp(-x) + 1,
            self.alpha * (torch.exp(x) - 1)
        )
        return out 

class CBT(nn.Module):
    def __init__(self, in_channels, out_channels=None, alpha=2.0):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = TinyReLU(alpha=alpha)

    def forward(self, x):
        x = self.conv(x)  # [B, out_channels, H, W]
        x = self.bn(x)    # [B, out_channels, H, W]
        x = self.act(x)   # [B, out_channels, H, W]
        return x 
    
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)
        
    def forward(self, x):
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # Depthwise Conv
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.norm2 = nn.LayerNorm(4 * dim, eps=1e-6)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = nn.Identity()  # 可替换为DropPath实现

    def forward(self, x):
        shortcut = x  # [B, C, H, W] # 保存残差连接的输入
        x = self.dwconv(x)  # [B, C, H, W] # 7x7深度可分离卷积,提取局部特征
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C] # 调整维度顺序以适配LayerNorm
        x = self.norm1(x)  # [B, H, W, C] # 第一次LayerNorm归一化
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W] # 恢复原始维度顺序
        x = self.pwconv1(x)  # [B, 4C, H, W] # 1x1逐点卷积,升维扩展通道数
        x = x.permute(0, 2, 3, 1)  # [B, H, W, 4C] # 调整维度顺序以适配LayerNorm
        x = self.norm2(x)  # [B, H, W, 4C] # 第二次LayerNorm归一化
        x = x.permute(0, 3, 1, 2)  # [B, 4C, H, W] # 恢复原始维度顺序
        x = self.pwconv2(x)  # [B, C, H, W] # 1x1逐点卷积,降维回原始通道数
        x = self.gamma.view(1, -1, 1, 1) * x  # [B, C, H, W] # 可学习的层缩放系数
        x = self.drop_path(x)  # [B, C, H, W] # DropPath正则化(此处为Identity)
        x = x + shortcut  # [B, C, H, W] # 残差连接
        return x

class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = LayerNorm2d(in_dim, eps=1e-6)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.norm(x)  # [B, C, H, W]
        x = self.conv(x)  # [B, out_dim, H/2, W/2]
        return x

class FeatureExtraction(nn.Module):
    def __init__(self, use_pretrained=True, model_variant='tiny', pretrained_path=None):
        super().__init__()
        
        # 设置预训练模型参数的维度
        if model_variant == 'tiny':
            dims = [96, 192, 384, 768]  # ConvNeXt-Tiny的各阶段通道数
        elif model_variant == 'small':
            dims = [96, 192, 384, 768]  # ConvNeXt-Small的各阶段通道数
        elif model_variant == 'base':
            dims = [128, 256, 512, 1024]  # ConvNeXt-Base的各阶段通道数
        else:
            raise ValueError(f"Unsupported model variant: {model_variant}")
        
        # 初始化self.use_pretrained，默认假设我们不会成功使用预训练模型
        self.use_pretrained = False
        self.model_variant = model_variant
        
        # 尝试加载预训练模型
        if use_pretrained:
            # 先尝试从本地加载预训练权重
            if pretrained_path and os.path.exists(pretrained_path):
                print(f"正在从本地加载预训练权重: {pretrained_path}")
                try:
                    # 先创建模型结构
                    if model_variant == 'tiny':
                        model = models.convnext_tiny(pretrained=False)
                    elif model_variant == 'small':
                        model = models.convnext_small(pretrained=False)
                    elif model_variant == 'base':
                        model = models.convnext_base(pretrained=False)
                    
                    # 加载本地权重
                    state_dict = torch.load(pretrained_path, map_location='cpu')
                    
                    # 如果权重是从huggingface下载的，可能需要处理键名前缀
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    
                    # 加载权重
                    model.load_state_dict(state_dict, strict=False)
                    print("预训练权重加载成功")
                    self.use_pretrained = True
                except Exception as e:
                    print(f"从本地加载预训练权重失败: {e}")
                    print("将使用随机初始化的权重")
                    
            # 如果没有从本地加载成功，尝试从网络下载
            if not self.use_pretrained:
                try:
                    print("尝试从网络下载预训练模型...")
                    if model_variant == 'tiny':
                        model = models.convnext_tiny(pretrained=True)
                    elif model_variant == 'small':
                        model = models.convnext_small(pretrained=True)
                    elif model_variant == 'base':
                        model = models.convnext_base(pretrained=True)
                    print("预训练模型下载成功")
                    self.use_pretrained = True
                except Exception as e:
                    print(f"从网络下载预训练模型失败: {e}")
                    print("将使用随机初始化的权重")
        
        # 如果成功加载了预训练模型
        if self.use_pretrained:
            # 提取ConvNeXt的各个阶段
            # ConvNeXt的结构: stem -> stages[0] -> stages[1] -> stages[2] -> stages[3]
            self.stem = model.features[0]  # 下采样4倍的stem
            self.stage1 = model.features[1]  # 第1阶段，输出特征图大小为输入的1/4
            self.stage2 = model.features[2]  # 第2阶段，输出特征图大小为输入的1/8
            self.stage3 = model.features[3]  # 第3阶段，输出特征图大小为输入的1/16
            self.stage4 = model.features[4]  # 第4阶段，输出特征图大小为输入的1/32
            
            # 适配层，将预训练模型的通道数映射到我们模型需要的通道数
            self.adapt1 = nn.Conv2d(dims[0], 128, kernel_size=1)
            self.adapt2 = nn.Conv2d(dims[1], 256, kernel_size=1)
            self.adapt3 = nn.Conv2d(dims[2], 512, kernel_size=1)
            self.adapt4 = nn.Conv2d(dims[3], 1024, kernel_size=1)
        else:
            # 使用原来的自定义实现
            self.stem = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=4, stride=4),  # [B, 3, H, W] -> [B, 128, H/4, W/4]
                LayerNorm2d(128, eps=1e-6) # [B, 128, H/4, W/4]
            )
            self.stage1 = nn.Sequential(*[ConvNeXtBlock(128) for _ in range(3)])  # [B, 128, H/4, W/4]
            self.down1 = Downsample(128, 256)  # [B, 256, H/8, W/8]
            self.stage2 = nn.Sequential(*[ConvNeXtBlock(256) for _ in range(3)])  # [B, 256, H/8, W/8]
            self.down2 = Downsample(256, 512)  # [B, 512, H/16, W/16]
            self.stage3 = nn.Sequential(*[ConvNeXtBlock(512) for _ in range(27)])  # [B, 512, H/16, W/16]
            self.down3 = Downsample(512, 1024)  # [B, 1024, H/32, W/32]
            self.stage4 = nn.Sequential(*[ConvNeXtBlock(1024) for _ in range(3)])  # [B, 1024, H/32, W/32]
        
        # 输出特征形状信息
        print(f"使用预训练模型: {self.use_pretrained}, 变体: {model_variant}")
        
    def forward(self, x):
        if self.use_pretrained:
            # 使用ConvNeXt预训练backbone
            x1 = self.stem(x)  # 1/4 分辨率
            x2 = self.stage1(x1)  # 1/4 分辨率
            x3 = self.stage2(x2)  # 1/8 分辨率
            x4 = self.stage3(x3)  # 1/16 分辨率
            x5 = self.stage4(x4)  # 1/32 分辨率
            
            # 通过适配层调整通道数
            feat1 = self.adapt1(x2)  # 对应于1/4分辨率
            feat2 = self.adapt2(x3)  # 对应于1/8分辨率
            feat3 = self.adapt3(x4)  # 对应于1/16分辨率
            feat4 = self.adapt4(x5)  # 对应于1/32分辨率
        else:
            # 使用原始实现
            x = self.stem(x)  # [B, 128, H/4, W/4]
            x = self.stage1(x)  # [B, 128, H/4, W/4]
            feat1 = x
            x = self.down1(x)  # [B, 256, H/8, W/8]
            x = self.stage2(x)  # [B, 256, H/8, W/8]
            feat2 = x
            x = self.down2(x)  # [B, 512, H/16, W/16]
            x = self.stage3(x)  # [B, 512, H/16, W/16]
            feat3 = x
            x = self.down3(x)  # [B, 1024, H/32, W/32]
            x = self.stage4(x)  # [B, 1024, H/32, W/32]
            feat4 = x
            
        return [feat1, feat2, feat3, feat4]  # 返回多尺度特征图 

class PixelClassifier(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x) 
    
class TBF(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.cfb_conv = nn.Conv2d(sum(in_channels_list), out_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        # 每个输入特征都用1x1卷积对齐到out_channels
        self.align_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])

    def forward(self, features):
        # features: list of [B, C_i, H_i, W_i]
        target_size = features[0].shape[2:]
        upsampled = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in features]
        cfb_input = torch.cat(upsampled, dim=1)
        cfb_out = self.cfb_conv(cfb_input)
        # SFB: 先通道对齐再相乘
        aligned = [conv(f) for conv, f in zip(self.align_convs, upsampled)]  # 每个[B, out_channels, H, W]
        sfb_out = aligned[0]
        for feat in aligned[1:]:
            sfb_out = sfb_out * feat
        out = self.sigmoid(cfb_out + self.final_conv(sfb_out))
        return out 

class TinyFeatureAmplification(nn.Module):
    def __init__(self, in_channels, dilation_rates=[1, 2, 3]):
        super().__init__()
        # 目标过滤阶段的条带卷积
        self.filter_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0)),
        ])
        # 目标放大阶段的膨胀卷积
        self.amp_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=rate, dilation=rate)
            for rate in dilation_rates
        ])
        self.sigmoid = nn.Sigmoid()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def background_suppression(self, x):
        # x: (B, C, H, W)
        # Y-Max
        y_max, _ = torch.max(x, dim=2, keepdim=True)  # (B, C, 1, W)
        y_max_expand = y_max.expand_as(x)  # (B, C, H, W)
        # X-Max
        x_max, _ = torch.max(x, dim=3, keepdim=True)  # (B, C, H, 1)
        x_max_expand = x_max.expand_as(x)  # (B, C, H, W)
        # Z-Max (通道最大)
        z_max, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        z_max_expand = z_max.repeat(1, x.shape[1], 1, 1)  # (B, C, H, W)
        z_max_conv = self.conv1x1(z_max_expand)  # (B, C, H, W)
        # 融合
        out = x + y_max_expand + x_max_expand + z_max_conv  # (B, C, H, W)
        return out

    def object_filtering(self, x):
        # 四个条带卷积+Sigmoid后相加
        filtered = [self.sigmoid(conv(x)) for conv in self.filter_convs]  # 每个(B, C, H, W)
        out = sum(filtered)  # (B, C, H, W)
        return out

    def object_amplification(self, x):
        # 多个膨胀卷积嵌套顺序执行
        out = x
        for conv in self.amp_convs:
            out = conv(out)  # (B, C, H, W)
        return out

    def forward(self, x):
        x = self.background_suppression(x)  # (B, C, H, W)
        x = self.object_filtering(x)        # (B, C, H, W)
        x = self.object_amplification(x)    # (B, C, H, W)
        return x 
    
class EAFNet(nn.Module):
    def __init__(self, 
                 backbone_channels=[128, 256, 512, 1024], 
                 tfa_dilation=[1, 2, 3], 
                 cbt_alpha=2.0,
                 use_pretrained=False,
                 model_variant='tiny',
                 pretrained_path=None):
        super().__init__()
        self.backbone = FeatureExtraction(use_pretrained=use_pretrained, 
                                         model_variant=model_variant,
                                         pretrained_path=pretrained_path)
        self.tfa = TinyFeatureAmplification(backbone_channels[-1], dilation_rates=tfa_dilation)
        self.cbt4 = CBT(512, 512, alpha=cbt_alpha)
        self.cbt3 = CBT(256, 256, alpha=cbt_alpha)
        self.cbt2 = CBT(128, 128, alpha=cbt_alpha)
        self.cbt1 = CBT(128, 128, alpha=cbt_alpha)
        self.cbt4_fusion = CBT(1024, 256, alpha=cbt_alpha)
        self.cbt3_fusion = CBT(512, 256, alpha=cbt_alpha)
        self.cbt2_fusion = CBT(256, 128, alpha=cbt_alpha)
        self.cbt1_fusion = CBT(128, 128, alpha=cbt_alpha)
        self.tbf = TBF([128, 128, 256, 256], 128)
        self.pixel_classifier = PixelClassifier(128, 1)
        self.align4to3 = nn.Conv2d(1024, 512, kernel_size=1)
        self.align3to2 = nn.Conv2d(512, 256, kernel_size=1)
        self.align2to1 = nn.Conv2d(256, 128, kernel_size=1)

    def forward(self, x):
        # 1. 特征提取
        F1, F2, F3, F4 = self.backbone(x)  #
        # F1: [B, 128, 128, 128]
        # F2: [B, 256, 64, 64]
        # F3: [B, 512, 32, 32]
        # F4: [B, 1024, 16, 16]

        # 2. TFA
        f4 = self.tfa(F4)  # [B, 1024, 16, 16]

        # 3. 放大路径（上采样后通道对齐）
        f3 = self.align4to3(F.interpolate(f4, size=F3.shape[2:], mode='bilinear', align_corners=False)) + self.cbt4(F3)  # [B, 512, 32, 32]
        f2 = self.align3to2(F.interpolate(f3, size=F2.shape[2:], mode='bilinear', align_corners=False)) + self.cbt3(F2)  # [B, 256, 64, 64]
        f1 = self.align2to1(F.interpolate(f2, size=F1.shape[2:], mode='bilinear', align_corners=False)) + self.cbt2(F1)  # [B, 128, 128, 128]

        # 4. 融合路径
        f1_ = self.cbt1_fusion(f1)   # [B, 128, 128, 128]
        f2_ = self.cbt2_fusion(f2)   # [B, 128, 64, 64]
        f3_ = self.cbt3_fusion(f3)   # [B, 256, 32, 32]
        f4_ = self.cbt4_fusion(f4)   # [B, 256, 16, 16]

        # 5. TBF融合（全部上采样到128x128，拼接后通道数128+128+256+256=768，输出128通道）
        f0 = self.tbf([f1_, f2_, f3_, f4_])  # [B, 128, 128, 128]

        # 6. 像素分类器
        out = self.pixel_classifier(f0)  # [B, 1, 128, 128]

        # 添加sigmoid
        out = torch.sigmoid(out)  # [B, 1, 128, 128]

        return out 
    
def test():
    x = torch.randn((1, 3, 512, 512)).cuda()  # batch size, channel,height,width

    model = EAFNet().cuda()
    flops, params = profile(model, (x,))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
    preds = model(x)
    print(x.shape)
    print(preds.shape)


if __name__ == "__main__":
    test()
