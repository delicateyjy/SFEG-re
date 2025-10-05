# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch


def average_performance(pred, target, thr=None, k=None):
    """计算 CP、CR、CF1、OP、OR、OF1，其中 C 代表每类
    平均值，O 代表整体平均值，P 代表精确度，R 代表
    召回率，F1 代表 F1 分数。

    参数：
        pred (torch.Tensor | np.ndarray)：模型预测，形状为
            (N, C)，其中 C 是类别数。
        target (torch.Tensor | np.ndarray)：每个预测的目标，形状为
            (N, C)，其中 C 是类别数。1 代表
            正例，0 代表负例，-1 代表
            困难例子。
        thr (float)：置信阈值。默认为 None。
        k (int)：Top-k 性能。请注意，如果同时给出 thr 和 k，则 k
            将被忽略。默认为 None。

    返回：
        tuple: (CP, CR, CF1, OP, OR, OF1)
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')
    if thr is None and k is None:
        thr = 0.5
        warnings.warn('Neither thr nor k is given, set thr as 0.5 by '
                      'default.')
    elif thr is not None and k is not None:
        warnings.warn('Both thr and k are given, use threshold in favor of '
                      'top-k.')

    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'

    eps = np.finfo(np.float32).eps
    target[target == -1] = 0
    if thr is not None:
        # a label is predicted positive if the confidence is no lower than thr
        pos_inds = pred >= thr

    else:
        # top-k labels will be predicted positive for any example
        sort_inds = np.argsort(-pred, axis=1)
        sort_inds_ = sort_inds[:, :k]
        inds = np.indices(sort_inds_.shape)
        pos_inds = np.zeros_like(pred)
        pos_inds[inds[0], sort_inds_] = 1

    tp = (pos_inds * target) == 1
    fp = (pos_inds * (1 - target)) == 1
    fn = ((1 - pos_inds) * target) == 1

    precision_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fp.sum(axis=0), eps)
    recall_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fn.sum(axis=0), eps)
    CP = precision_class.mean() * 100.0
    CR = recall_class.mean() * 100.0
    CF1 = 2 * CP * CR / np.maximum(CP + CR, eps)
    OP = tp.sum() / np.maximum(tp.sum() + fp.sum(), eps) * 100.0
    OR = tp.sum() / np.maximum(tp.sum() + fn.sum(), eps) * 100.0
    OF1 = 2 * OP * OR / np.maximum(OP + OR, eps)
    return CP, CR, CF1, OP, OR, OF1
