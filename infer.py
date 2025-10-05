
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.DataProcessing import RoadDataset
from models.SFEG import SFEG
from torchvision import transforms
from PIL import Image
import os

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
to_pil = transforms.ToPILImage()


# 参数设置
# imgsz_h/imgsz_w为None时保持原分辨率，否则自定义分辨率
imgsz_h = 512  # 例如256，或None保持原高
imgsz_w = 512  # 例如256，或None保持原宽
batchsz = 1

# 推理图片文件夹路径（可修改为你的图片文件夹）
ImgPath = "C:/Users/yejia/Desktop/CDS-Net-master/dataset/cracktree200/val/"  # 需要推理的图片路径
resultsdir = "results/cracktree200/val/"  # 结果保存路径

# 检查推理图片路径是否存在
if not os.path.exists(ImgPath):
        raise FileNotFoundError(f"推理图片路径不存在: {ImgPath}")

# 创建结果保存目录
os.makedirs(resultsdir, exist_ok=True)


# 创建数据集和加载器（支持自定义或原分辨率）
InferenceSet = RoadDataset(ImgPath, ImgPath, imgsz_h, imgsz_w, augment=False)
InferenceLoader = DataLoader(InferenceSet, batch_size=batchsz, shuffle=False)

# 加载模型
Network = SFEG(embed_dim=[32, 64, 128, 256, 512], depth=[3, 3, 3, 3, 3])
model_path = './best_cracktree200_final.pth'
if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重文件不存在: {model_path}")
Network.load_state_dict(torch.load(model_path, map_location=DEVICE))
Network = Network.to(DEVICE)
Network.eval()

# 推理过程


with torch.no_grad():
        for idx, samples in enumerate(tqdm(InferenceLoader, desc='推理进度')):
                img = samples['image']
                padding = samples['padding'][0] if 'padding' in samples else (0, 0, 0, 0)
                orig_size = samples['orig_size'][0] if 'orig_size' in samples else None
                img = img.to(DEVICE)
                # SFEG模型返回3个值，取第一个为mask
                mask, _, _ = Network(img)
                mask = torch.sigmoid(mask)
                pred = (mask >= 0.5).float()
                # 还原为原图尺寸
                pred_img = pred.squeeze(0).cpu()
                pred_pil = to_pil(pred_img)
                # 裁剪掉pad
                if padding != (0, 0, 0, 0):
                        w, h = pred_pil.size
                        left, top, right, bottom = padding
                        pred_pil = pred_pil.crop((left, top, w - right, h - bottom))
                # 再resize回原始尺寸
                if orig_size is not None:
                        pred_pil = pred_pil.resize(orig_size, Image.NEAREST)
                pred_pil.save(os.path.join(resultsdir, f'img{idx+1}_pred.png'))

print('Inference completed! Results saved in results directory.')
