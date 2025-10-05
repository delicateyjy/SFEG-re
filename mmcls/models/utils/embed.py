# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import AdaptivePadding
from mmcv.runner.base_module import BaseModule

from .helpers import to_2tuple


def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """调整 pos_embed 权重。

    参数：
        pos_embed (torch.Tensor)：形状为 [1, L, C] 的位置嵌入权重。
        src_shape (tuple)：降采样的原始训练图像的分辨率，格式为 (H, W)。
        dst_shape (tuple)：降采样的新训练图像的分辨率，格式为 (H, W)。
        mode (str)：用于上采样的算法。选择 'nearest'、'linear'、'bilinear'、'bicubic' 和 'trilinear' 中的一个。
            默认为 'bicubic'。
        num_extra_tokens (int)：额外标记的数量，例如 cls_token。
            默认为 1。

    返回：
        torch.Tensor：形状为 [1, L_new, C] 的调整大小后的 pos_embed。
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    dst_weight = F.interpolate(
        src_weight, size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)

    return torch.cat((extra_tokens, dst_weight), dim=1)


def resize_relative_position_bias_table(src_shape, dst_shape, table, num_head):
    """调整相对位置偏置表。

    参数：
        src_shape (int)：下采样的原始训练图像的分辨率，格式为 (H, W)。 
        dst_shape (int)：下采样的新训练图像的分辨率，格式为 (H, W)。 
        table (tensor)：预训练模型的相对位置偏置。 
        num_head (int)：注意力头的数量。 

    返回：
        torch.Tensor：调整后的相对位置偏置表。
    """
    from scipy import interpolate

    def geometric_progression(a, r, n):
        return a * (1.0 - r**n) / (1.0 - r)

    left, right = 1.01, 1.5
    while right - left > 1e-6:
        q = (left + right) / 2.0
        gp = geometric_progression(1, q, src_shape // 2)
        if gp > dst_shape // 2:
            right = q
        else:
            left = q

    dis = []
    cur = 1
    for i in range(src_shape // 2):
        dis.append(cur)
        cur += q**(i + 1)

    r_ids = [-_ for _ in reversed(dis)]

    x = r_ids + [0] + dis
    y = r_ids + [0] + dis

    t = dst_shape // 2.0
    dx = np.arange(-t, t + 0.1, 1.0)
    dy = np.arange(-t, t + 0.1, 1.0)

    all_rel_pos_bias = []

    for i in range(num_head):
        z = table[:, i].view(src_shape, src_shape).float().numpy()
        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
        all_rel_pos_bias.append(
            torch.Tensor(f_cubic(dx,
                                 dy)).contiguous().view(-1,
                                                        1).to(table.device))
    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
    return new_rel_pos_bias


class PatchEmbed(BaseModule):
    """图像到补丁嵌入。

    我们使用卷积层来实现PatchEmbed。

    参数：
        img_size（int | tuple）：输入图像的大小。默认：224
        in_channels（int）：输入通道的数量。默认：3
        embed_dims（int）：嵌入的维度。默认：768
        norm_cfg（dict，可选）：归一化层的配置字典。
            默认：无
        conv_cfg（dict，可选）：卷积层的配置字典。
            默认：无
        init_cfg（`mmcv.ConfigDict`，可选）：初始化的配置。
            默认：无
    """

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 embed_dims=768,
                 norm_cfg=None,
                 conv_cfg=None,
                 init_cfg=None):
        super(PatchEmbed, self).__init__(init_cfg)
        warnings.warn('The `PatchEmbed` in mmcls will be deprecated. '
                      'Please use `mmcv.cnn.bricks.transformer.PatchEmbed`. '
                      "It's more general and supports dynamic input shape")

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
        self.embed_dims = embed_dims

        # Use conv layer to embed
        conv_cfg = conv_cfg or dict()
        _conv_cfg = dict(
            type='Conv2d', kernel_size=16, stride=16, padding=0, dilation=1)
        _conv_cfg.update(conv_cfg)
        self.projection = build_conv_layer(_conv_cfg, in_channels, embed_dims)

        # Calculate how many patches a input image is splited to.
        h_out, w_out = [(self.img_size[i] + 2 * self.projection.padding[i] -
                         self.projection.dilation[i] *
                         (self.projection.kernel_size[i] - 1) - 1) //
                        self.projection.stride[i] + 1 for i in range(2)]

        self.patches_resolution = (h_out, w_out)
        self.num_patches = h_out * w_out

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't " \
            f'match model ({self.img_size[0]}*{self.img_size[1]}).'
        # The output size is (B, N, D), where N=H*W/P/P, D is embid_dim
        x = self.projection(x).flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x


# Modified from pytorch-image-models
class HybridEmbed(BaseModule):
    """CNN特征图嵌入。

    从CNN中提取特征图，扁平化，
    投影到嵌入维度。

    参数：
        backbone (nn.Module)：CNN主干
        img_size (int | tuple)：输入图像的大小。默认：224
        feature_size (int | tuple, 可选)：由CNN主干提取的特征图大小。默认：无
        in_channels (int)：输入通道的数量。默认：3
        embed_dims (int)：嵌入的维度。默认：768
        conv_cfg (dict, 可选)：用于卷积层的配置字典。默认：无。
        init_cfg (`mmcv.ConfigDict`, 可选)：初始化的配置。默认：无。
    """

    def __init__(self,
                 backbone,
                 img_size=224,
                 feature_size=None,
                 in_channels=3,
                 embed_dims=768,
                 conv_cfg=None,
                 init_cfg=None):
        super(HybridEmbed, self).__init__(init_cfg)
        assert isinstance(backbone, nn.Module)
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of
                #  determining the exact dim of the output feature
                #  map for all networks, the feature metadata has
                #  reliable channel and stride info, but using
                #  stride to calc feature dim requires info about padding of
                #  each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(
                    torch.zeros(1, in_channels, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    # last feature if backbone outputs list/tuple of features
                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]

        # Use conv layer to embed
        conv_cfg = conv_cfg or dict()
        _conv_cfg = dict(
            type='Conv2d', kernel_size=1, stride=1, padding=0, dilation=1)
        _conv_cfg.update(conv_cfg)
        self.projection = build_conv_layer(_conv_cfg, feature_dim, embed_dims)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            # last feature if backbone outputs list/tuple of features
            x = x[-1]
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


class PatchMerging(BaseModule):
    """合并补丁特征图。修改自 mmcv，使用的是前归一化层，而 Swin V2 在这里使用后归一化。因此，添加额外参数以决定是否使用后归一化。

    该层根据 kernel_size 对特征图进行分组，并将归一化和线性层应用于分组后的特征图（在 Swin Transformer 中使用）。我们的实现使用 `nn.Unfold` 来合并补丁，比原始实现快约 25%。但是，我们需要修改预训练模型以兼容。

    参数：
        in_channels (int): 输入通道的数量。
            以完全覆盖您指定的过滤器和步幅。
        out_channels (int): 输出通道的数量。
        kernel_size (int | tuple, optional): 展开层中的卷积核大小。
            默认值为 2。
        stride (int | tuple, optional): 展开层中滑动块的步幅。
            默认值为 None。（将设置为 `kernel_size`）
        padding (int | tuple | string): 嵌入卷积的填充长度。当它是字符串时，表示自适应填充模式，当前支持“same”和“corner”。
            默认值为“corner”。
        dilation (int | tuple, optional): 展开层中的膨胀参数。默认值：1。
        bias (bool, optional): 是否在线性层中添加偏置。
            默认值为 False。
        norm_cfg (dict, optional): 归一化层的配置字典。
            默认值为 dict(type='LN')。
        is_post_norm (bool): 是否在这里使用后归一化。
            默认值为 False。
        init_cfg (dict, optional): 初始化的额外配置。
            默认值为 None。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 is_post_norm=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_post_norm = is_post_norm

        if stride:
            stride = stride
        else:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of unfold
            padding = 0
        else:
            self.adaptive_padding = None

        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

        if norm_cfg is not None:
            # build pre or post norm layer based on different channels
            if self.is_post_norm:
                self.norm = build_norm_layer(norm_cfg, out_channels)[1]
            else:
                self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

    def forward(self, x, input_size):
        """
        参数：
            x（张量）：形状为（B，H*W，C_in）。
            input_size（元组[int]）：x的空间形状，排列为（H，W）。
                默认值：无。

        返回：
            元组：包含合并结果及其空间形状。

            - x（张量）：形状为（B，Merged_H * Merged_W，C_out）
            - out_size（元组[int]）：x的空间形状，排列为
              （Merged_H，Merged_W）。
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W

        if self.adaptive_padding:
            x = self.adaptive_padding(x)
            H, W = x.shape[-2:]

        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)
        x = self.sampler(x)

        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] *
                 (self.sampler.kernel_size[0] - 1) -
                 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] *
                 (self.sampler.kernel_size[1] - 1) -
                 1) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C

        if self.is_post_norm:
            # use post-norm here
            x = self.reduction(x)
            x = self.norm(x) if self.norm else x
        else:
            x = self.norm(x) if self.norm else x
            x = self.reduction(x)

        return x, output_size
