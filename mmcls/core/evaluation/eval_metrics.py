# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number

import numpy as np
import torch
from torch.nn.functional import one_hot


def calculate_confusion_matrix(pred, target):
    """根据预测和目标计算混淆矩阵。

    参数：
        pred (torch.Tensor | np.array)：模型预测，形状为 (N, C)。
        target (torch.Tensor | np.array)：每个预测的目标，形状为 (N, 1) 或 (N,)。

    返回：
        torch.Tensor：混淆矩阵
            形状为 (C, C)，其中 C 是类别的数量。
    """

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert (
        isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    # Modified from PyTorch-Ignite
    num_classes = pred.size(1)
    pred_label = torch.argmax(pred, dim=1).flatten()
    target_label = target.flatten()
    assert len(pred_label) == len(target_label)

    with torch.no_grad():
        indices = num_classes * target_label + pred_label
        matrix = torch.bincount(indices, minlength=num_classes**2)
        matrix = matrix.reshape(num_classes, num_classes)
    return matrix


def precision_recall_f1(pred, target, average_mode='macro', thrs=0.):
    """根据预测和目标计算精确度、召回率和F1分数。

    参数：
        pred (torch.Tensor | np.array)：模型预测，形状为 (N, C)。
        target (torch.Tensor | np.array)：每个预测的目标，形状为 (N, 1) 或 (N,).
        average_mode (str)：对结果进行的平均类型。
            选项为 'macro' 和 'none'。如果是 'none'，则返回每个类别的分数。如果是 'macro'，则计算每个类别的指标，并找到它们的未加权平均值。
            默认为 'macro'。
        thrs (Number | tuple[Number], 可选)：预测得分低于阈值的被视为负。默认为 0。

    返回：
        tuple：包含精确度、召回率、F1分数的元组。

        精确度、召回率、F1分数的类型为以下之一：

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    assert isinstance(pred, torch.Tensor), \
        (f'pred should be torch.Tensor or np.ndarray, but got {type(pred)}.')
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).long()
    assert isinstance(target, torch.Tensor), \
        f'target should be torch.Tensor or np.ndarray, ' \
        f'but got {type(target)}.'

    if isinstance(thrs, Number):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    num_classes = pred.size(1)
    pred_score, pred_label = torch.topk(pred, k=1)
    pred_score = pred_score.flatten()
    pred_label = pred_label.flatten()

    gt_positive = one_hot(target.flatten(), num_classes)

    precisions = []
    recalls = []
    f1_scores = []
    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        pred_positive = one_hot(pred_label, num_classes)
        if thr is not None:
            pred_positive[pred_score <= thr] = 0
        class_correct = (pred_positive & gt_positive).sum(0)
        precision = class_correct / np.maximum(pred_positive.sum(0), 1.) * 100
        recall = class_correct / np.maximum(gt_positive.sum(0), 1.) * 100
        f1_score = 2 * precision * recall / np.maximum(
            precision + recall,
            torch.finfo(torch.float32).eps)
        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            f1_score = float(f1_score.mean())
        elif average_mode == 'none':
            precision = precision.detach().cpu().numpy()
            recall = recall.detach().cpu().numpy()
            f1_score = f1_score.detach().cpu().numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    if return_single:
        return precisions[0], recalls[0], f1_scores[0]
    else:
        return precisions, recalls, f1_scores


def precision(pred, target, average_mode='macro', thrs=0.):
    """根据预测和目标计算精确度。

    参数：
        pred (torch.Tensor | np.array)：模型预测，形状为 (N, C)。
        target (torch.Tensor | np.array)：每个预测的目标，形状为 (N, 1) 或 (N,)。 
        average_mode (str)：对结果进行的平均类型。
            选项为 'macro' 和 'none'。如果是 'none'，则返回每个类别的分数。如果是 'macro'，则计算每个类别的指标，并找到它们的未加权平均值。
            默认值为 'macro'。
        thrs (Number | tuple[Number], optional)：分数低于阈值的预测被视为负类。默认为 0。

    返回：
         float | np.array | list[float | np.array]：精确度。

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    precisions, _, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return precisions


def recall(pred, target, average_mode='macro', thrs=0.):
    """根据预测和目标计算召回率。

    参数：
        pred (torch.Tensor | np.array)：模型预测，形状为 (N, C)。
        target (torch.Tensor | np.array)：每个预测的目标，形状为 (N, 1) 或 (N,).
        average_mode (str)：对结果执行的平均类型。
            选项为 'macro' 和 'none'。如果是 'none'，则返回每个类别的分数。如果是 'macro'，则计算每个类别的指标，并找到它们的无权平均值。
            默认值为 'macro'。
        thrs (Number | tuple[Number], 可选)：分数低于阈值的预测被视为负。
            默认值为 0。

    返回：
         float | np.array | list[float | np.array]：召回率。

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    _, recalls, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return recalls


def f1_score(pred, target, average_mode='macro', thrs=0.):
    """根据预测和目标计算F1分数。

    参数：
        pred (torch.Tensor | np.array)：模型预测，形状为(N, C)。
        target (torch.Tensor | np.array)：每个预测的目标，形状为(N, 1)或(N,)。
        average_mode (str)：对结果执行的平均类型。
            选项为'macro'和'none'。如果'none'，则返回每个类别的分数。如果'macro'，则计算每个类别的指标，并找到它们的无权重平均值。
            默认为'macro'。
        thrs (Number | tuple[Number], 可选)：得分低于阈值的预测被视为负。默认为0。

    返回：
         float | np.array | list[float | np.array]：F1分数。

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    _, _, f1_scores = precision_recall_f1(pred, target, average_mode, thrs)
    return f1_scores


def support(pred, target, average_mode='macro'):
    """根据预测和目标计算每个标签的总出现次数。

    参数：
        pred (torch.Tensor | np.array)：模型预测，形状为 (N, C)。
        target (torch.Tensor | np.array)：每个预测的目标，形状为 (N, 1) 或 (N,)，
        average_mode (str)：对结果进行平均的类型。
            选项有 'macro' 和 'none'。如果为 'none'，则返回每个类别的分数。如果为 'macro'，则计算每个类别的指标，并找到它们的未加权总和。
            默认值为 'macro'。

    返回：
        float | np.array：支持度。

            - 如果 ``average_mode`` 设置为 macro，函数返回一个单一的浮点数。
            - 如果 ``average_mode`` 设置为 none，函数返回一个形状为 C 的 np.array。
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.sum(1)
        if average_mode == 'macro':
            res = float(res.sum().numpy())
        elif average_mode == 'none':
            res = res.numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
    return res
