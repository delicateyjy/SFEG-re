import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    图像Patch嵌入模块
    将图像分块并转换为嵌入向量
    输入: (B, C, H, W)
    输出: (B, embed_dim, H', W')
    """
    def __init__(self, in_chans=3, patch_size=1, embed_dim=32, kernel_size=3):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=kernel_size, 
            stride=patch_size,
            padding=(kernel_size - patch_size + 1) // 2
        )

    def forward(self, x):
        return self.proj(x)
    