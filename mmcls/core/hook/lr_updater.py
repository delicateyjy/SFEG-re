# Copyright (c) OpenMMLab. All rights reserved.
from math import cos, pi

from mmcv.runner.hooks import HOOKS, LrUpdaterHook


@HOOKS.register_module()
class CosineAnnealingCooldownLrUpdaterHook(LrUpdaterHook):
    """余弦退火学习率调度器，带冷却时间。

    参数：
        min_lr（浮点数，可选）：退火后的最小学习率。
            默认值为 None。
        min_lr_ratio（浮点数，可选）：退火后的最小学习比率。
            默认值为 None。
        cool_down_ratio（浮点数）：冷却比率。默认值为 0.1。
        cool_down_time（整数）：冷却时间。默认值为 10。
        by_epoch（布尔值）：如果为 True，学习率按 epochs 变化。如果为 False，学习率按 iterations 变化。默认值为 True。
        warmup（字符串，可选）：使用的预热类型。可以是 None（不使用预热）、'constant'、'linear' 或 'exp'。默认值为 None。
        warmup_iters（整数）：预热持续的迭代或 epoch 数量。默认值为 0。
        warmup_ratio（浮点数）：预热开始时使用的学习率等于 ``warmup_ratio * initial_lr``。默认值为 0.1。
        warmup_by_epoch（布尔值）：如果为 True，``warmup_iters`` 表示预热持续的 epoch 数量，否则表示预热持续的迭代数量。默认值为 False。

    注意：
        你需要设置 ``min_lr`` 和 ``min_lr_ratio`` 中的一个且仅一个。
    """

    def __init__(self,
                 min_lr=None,
                 min_lr_ratio=None,
                 cool_down_ratio=0.1,
                 cool_down_time=10,
                 **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.cool_down_time = cool_down_time
        self.cool_down_ratio = cool_down_ratio
        super(CosineAnnealingCooldownLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        if progress > max_progress - self.cool_down_time:
            return target_lr * self.cool_down_ratio
        else:
            max_progress = max_progress - self.cool_down_time

        return annealing_cos(base_lr, target_lr, progress / max_progress)


def annealing_cos(start, end, factor, weight=1):
    """计算退火余弦学习率。

    余弦退火从 `weight * start + (1 - weight) * end` 到 `end`，当百分比从 0.0 变为 1.0。

    参数：
        start (float): 余弦退火的起始学习率。
        end (float): 余弦退火的结束学习率。
        factor (float): 计算当前百分比时的 `pi` 系数。范围从 0.0 到 1.0。
        weight (float, 可选): 计算实际起始学习率时 `start` 和 `end` 的组合因子。默认为 1。
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out
