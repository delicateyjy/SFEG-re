# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/facebookresearch/pycls/blob/f8cd962737e33ce9e19b3083a33551da95c2d9c0/pycls/core/net.py  # noqa: E501
# Original licence: Copyright (c) 2019 Facebook, Inc under the Apache License 2.0  # noqa: E501

import itertools
import logging
from typing import List, Optional

import mmcv
import torch
import torch.nn as nn
from mmcv.runner import EpochBasedRunner, get_dist_info
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import print_log
from torch.functional import Tensor
from torch.nn import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.utils.data import DataLoader


def scaled_all_reduce(tensors: List[Tensor], num_gpus: int) -> List[Tensor]:
    """执行提供的张量的缩放全规约操作。

    输入张量在原地修改。目前仅支持求和规约操作。规约值按进程组的逆大小进行缩放。

    参数：
        tensors (List[torch.Tensor]): 要处理的张量。
        num_gpus (int): 要使用的GPU数量
    返回：
        List[torch.Tensor]: 处理后的张量。
    """
    # There is no need for reduction in the single-proc case
    if num_gpus == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / num_gpus)
    return tensors


@torch.no_grad()
def update_bn_stats(model: nn.Module,
                    loader: DataLoader,
                    num_samples: int = 8192,
                    logger: Optional[logging.Logger] = None) -> None:
    """计算训练数据上的精确BN统计信息。

    参数：
        model (nn.module)：将重新计算其bn统计信息的模型。
        loader (DataLoader)：PyTorch数据加载器。
        num_samples (int)：用于更新bn统计信息的样本数量。
            默认值为8192。
        logger (:obj:`logging.Logger` | None)：用于记录的日志记录器。
            默认值：None。
    """
    # get dist info
    rank, world_size = get_dist_info()
    # Compute the number of mini-batches to use, if the size of dataloader is
    # less than num_iters, use all the samples in dataloader.
    num_iter = num_samples // (loader.batch_size * world_size)
    num_iter = min(num_iter, len(loader))
    # Retrieve the BN layers
    bn_layers = [
        m for m in model.modules()
        if m.training and isinstance(m, (_BatchNorm))
    ]

    if len(bn_layers) == 0:
        print_log('No BN found in model', logger=logger, level=logging.WARNING)
        return
    print_log(
        f'{len(bn_layers)} BN found, run {num_iter} iters...', logger=logger)

    # Finds all the other norm layers with training=True.
    other_norm_layers = [
        m for m in model.modules()
        if m.training and isinstance(m, (_InstanceNorm, GroupNorm))
    ]
    if len(other_norm_layers) > 0:
        print_log(
            'IN/GN stats will not be updated in PreciseHook.',
            logger=logger,
            level=logging.INFO)

    # Initialize BN stats storage for computing
    # mean(mean(batch)) and mean(var(batch))
    running_means = [torch.zeros_like(bn.running_mean) for bn in bn_layers]
    running_vars = [torch.zeros_like(bn.running_var) for bn in bn_layers]
    # Remember momentum values
    momentums = [bn.momentum for bn in bn_layers]
    # Set momentum to 1.0 to compute BN stats that reflect the current batch
    for bn in bn_layers:
        bn.momentum = 1.0
    # Average the BN stats for each BN layer over the batches
    if rank == 0:
        prog_bar = mmcv.ProgressBar(num_iter)

    for data in itertools.islice(loader, num_iter):
        model.train_step(data)
        for i, bn in enumerate(bn_layers):
            running_means[i] += bn.running_mean / num_iter
            running_vars[i] += bn.running_var / num_iter
        if rank == 0:
            prog_bar.update()

    # Sync BN stats across GPUs (no reduction if 1 GPU used)
    running_means = scaled_all_reduce(running_means, world_size)
    running_vars = scaled_all_reduce(running_vars, world_size)
    # Set BN stats and restore original momentum values
    for i, bn in enumerate(bn_layers):
        bn.running_mean = running_means[i]
        bn.running_var = running_vars[i]
        bn.momentum = momentums[i]


@HOOKS.register_module()
class PreciseBNHook(Hook):
    """精确的 BN 钩子。

    重新计算并更新批归一化统计数据，以使其更加精确。在训练过程中，BN 统计数据和权重在每次迭代后都在变化，因此运行平均值无法精确反映当前模型的实际统计数据。

    通过这个钩子，BN 统计数据在固定权重下重新计算，以使运行平均值更加精确。具体而言，它计算每批次的均值/方差的真实平均值，而不是运行平均值。有关详细信息，请参见论文《重新思考批归一化中的批次》第 3 节。

    这个钩子将更新 BN 统计数据，因此应在 ``CheckpointHook`` 和 ``EMAHook`` 之前执行，通常将其优先级设置为 "ABOVE_NORMAL"。

    参数：
        num_samples (int)：用于更新 BN 统计数据的样本数量。默认为 8192。
        interval (int)：执行精确的 BN 间隔。默认为 1。
    """

    def __init__(self, num_samples: int = 8192, interval: int = 1) -> None:
        assert interval > 0 and num_samples > 0

        self.interval = interval
        self.num_samples = num_samples

    def _perform_precise_bn(self, runner: EpochBasedRunner) -> None:
        print_log(
            f'Running Precise BN for {self.num_samples} items...',
            logger=runner.logger)
        update_bn_stats(
            runner.model,
            runner.data_loader,
            self.num_samples,
            logger=runner.logger)
        print_log('Finish Precise BN, BN stats updated.', logger=runner.logger)

    def after_train_epoch(self, runner: EpochBasedRunner) -> None:
        """计算精确的批量归一化，并在GPU之间广播批量归一化统计信息。

        参数：
            runner (obj:`EpochBasedRunner`): 运行器对象。
        """
        assert isinstance(runner, EpochBasedRunner), \
            'PreciseBN only supports `EpochBasedRunner` by now'

        # if by epoch, do perform precise every `self.interval` epochs;
        if self.every_n_epochs(runner, self.interval):
            self._perform_precise_bn(runner)
