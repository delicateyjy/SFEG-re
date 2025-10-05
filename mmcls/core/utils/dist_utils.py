# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import OptimizerHook, get_dist_info
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)

from mmcls.utils import auto_select_device


def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)


def allreduce_grads(params, coalesce=True, bucket_size_mb=-1):
    grads = [
        param.grad.data for param in params
        if param.requires_grad and param.grad is not None
    ]
    world_size = dist.get_world_size()
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))


class DistOptimizerHook(OptimizerHook):

    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()


def sync_random_seed(seed=None, device=None):
    """确保不同的等级共享相同的种子。

所有工作者必须调用此函数，否则将导致死锁。
此方法通常用于 `DistributedSampler`，
因为种子在分布式组的所有进程中应是相同的。

在分布式采样中，不同的等级应从数据集中采样不重叠的数据。因此，此函数用于确保每个等级根据相同的种子以相同的顺序对数据索引进行洗牌。然后不同的等级可以使用不同的索引从相同的数据列表中选择不重叠的数据。

参数：
    seed（int，选填）：种子。默认为 None。
    device（str）：种子将放置的设备。
        默认为 'cuda'。

返回：
    int：要使用的种子。
    """
    if device is None:
        device = auto_select_device()
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()
