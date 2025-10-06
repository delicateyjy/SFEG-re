import torch
import torch.nn as nn

def get_gradient_prior(tensor):
    """
    提取梯度先验
    使用Sobel算子计算图像梯度幅值
    参数:
        tensor: (B, C, H, W) 输入张量
    返回:
        grad_magnitude: (B, 1, H, W) 梯度幅值图
    """
    # 多通道时取均值
    if tensor.dim() == 4 and tensor.size(1) > 1:
        tensor = tensor.mean(dim=1, keepdim=True)
    
    # Sobel算子
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
        dtype=torch.float32
    ).reshape(1, 1, 3, 3).to(tensor.device)
    
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
        dtype=torch.float32
    ).reshape(1, 1, 3, 3).to(tensor.device)
    
    # 计算梯度
    grad_x = nn.functional.conv2d(tensor, sobel_x, padding=1)
    grad_y = nn.functional.conv2d(tensor, sobel_y, padding=1)
    
    # 梯度幅值
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    
    return grad_magnitude