"""
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = StepLR(optimizer, step_size=2, gamma=0.96)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn import init
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from thop import profile

