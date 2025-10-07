'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

from typing import Iterable #用于标注 data_loader 参数类型。
import torch
import time
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                     epoch: int, args = None, logger = None):
    model.train()
    criterion.train()

    pbar = tqdm(total=len(data_loader.dataloader), desc=f"Initial Loss: Pending")
    for i, data in enumerate(data_loader):
        samples = data['image'].to(torch.device(args.device))
        targets = data['label'].to(torch.device(args.device))

        Mask1, Mask2, Mask3 = model(samples)
        Mask1 = Mask1.float()
        Mask2 = Mask2.float()
        Mask3 = Mask3.float()

        # targets[targets > 0] = 1

        STerm1 = criterion(Mask1, targets.float())
        STerm2 = criterion(Mask2, targets.float())
        STerm3 = criterion(Mask3, targets.float())
        loss_final = STerm1 + 0.5 * STerm2 + 0.5 * STerm3

        cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))

        loss_final_str = '{:.4f}'.format(loss_final.item())
        l = optimizer.param_groups[0]['lr']
        logger.info(f"time -> {cur_time} | Epoch -> {epoch} | image_num -> {data['A_paths']} | loss final -> {loss_final_str} | lr -> {l}")

        pbar.set_description(f"Loss: {loss_final.item():.4f}")
        pbar.update(1)
        optimizer.zero_grad()
        loss_final.backward()
        optimizer.step()

    pbar.close()
