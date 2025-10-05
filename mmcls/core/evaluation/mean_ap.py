# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch


def average_precision(pred, target):
    r"""计算单个类别的平均精度。

    AP 将精度-召回曲线总结为在任何 r'>r 下获得的最大精度的加权平均值，其中 r 是召回率：

    .. math::
        \text{AP} = \sum_n (R_n - R_{n-1}) P_n

    请注意，由于曲线是分段常数，因此没有涉及近似。

    参数：
        pred (np.ndarray)：形状为 (N, ) 的模型预测。
        target (np.ndarray)：形状为 (N, ) 的每个预测的目标。

    返回：
        float：作为平均精度值的单个浮点数。
    """
    eps = np.finfo(np.float32).eps

    # sort examples
    sort_inds = np.argsort(-pred)
    sort_target = target[sort_inds]

    # count true positive examples
    pos_inds = sort_target == 1
    tp = np.cumsum(pos_inds)
    total_pos = tp[-1]

    # count not difficult examples
    pn_inds = sort_target != -1
    pn = np.cumsum(pn_inds)

    tp[np.logical_not(pos_inds)] = 0
    precision = tp / np.maximum(pn, eps)
    ap = np.sum(precision) / np.maximum(total_pos, eps)
    return ap


def mAP(pred, target):
    """计算与类别相关的平均精度均值。

    参数：
        pred (torch.Tensor | np.ndarray)：模型预测，形状为
            (N, C)，其中 C 是类别的数量。
        target (torch.Tensor | np.ndarray)：每个预测的目标，形状为
            (N, C)，其中 C 是类别的数量。1 代表
            正例，0 代表负例，-1 代表
            困难例子。

    返回：
        float：一个浮点数作为 mAP 值。
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')

    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'
    num_classes = pred.shape[1]
    ap = np.zeros(num_classes)
    for k in range(num_classes):
        ap[k] = average_precision(pred[:, k], target[:, k])
    mean_ap = ap.mean() * 100.0
    return mean_ap
