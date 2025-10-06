import torch.nn as nn

class DownSample(nn.Module):
    """
    下采样模块
    通道减半 + 空间2倍下采样
    """
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 2, 
                     kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)  # 空间减半，通道翻倍
        )
    
    def forward(self, x):
        return self.proj(x)
    
class DownSample_CP(nn.Module):
    """
    下采样模块（用于Mask）
    空间2倍下采样 + 通道调整
    """
    def __init__(self, input_dim=4, output_dim=1, kernel_size=3, stride=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(input_dim, output_dim, kernel_size, stride, 
                     padding=1, bias=False)
        )
        
    def forward(self, x):
        return self.proj(x)