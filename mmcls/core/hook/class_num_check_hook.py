# Copyright (c) OpenMMLab. All rights reserved
from mmcv.runner import IterBasedRunner
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import is_seq_of


@HOOKS.register_module()
class ClassNumCheckHook(Hook):

    def _check_head(self, runner, dataset):
        """检查头部中的 `num_classes` 是否与 `dataset` 中的 `CLASSES` 的长度匹配。

        参数：
            runner (obj:`EpochBasedRunner`, `IterBasedRunner`): runner 对象。
            dataset (obj: `BaseDataset`): 需要检查的数据集。
        """
        model = runner.model
        if dataset.CLASSES is None:
            runner.logger.warning(
                f'Please set `CLASSES` '
                f'in the {dataset.__class__.__name__} and'
                f'check if it is consistent with the `num_classes` '
                f'of head')
        else:
            assert is_seq_of(dataset.CLASSES, str), \
                (f'`CLASSES` in {dataset.__class__.__name__}'
                 f'should be a tuple of str.')
            for name, module in model.named_modules():
                if hasattr(module, 'num_classes'):
                    assert module.num_classes == len(dataset.CLASSES), \
                        (f'The `num_classes` ({module.num_classes}) in '
                         f'{module.__class__.__name__} of '
                         f'{model.__class__.__name__} does not matches '
                         f'the length of `CLASSES` '
                         f'{len(dataset.CLASSES)}) in '
                         f'{dataset.__class__.__name__}')

    def before_train_iter(self, runner):
        """检查训练数据集是否与头部兼容。

        参数：
            runner（对象：`IterBasedRunner`）：基于迭代的运行器。
        """
        if not isinstance(runner, IterBasedRunner):
            return
        self._check_head(runner, runner.data_loader._dataloader.dataset)

    def before_val_iter(self, runner):
        """检查评估数据集是否与头部兼容。

        参数：
            runner (obj:`IterBasedRunner`): 基于迭代的运行器。
        """
        if not isinstance(runner, IterBasedRunner):
            return
        self._check_head(runner, runner.data_loader._dataloader.dataset)

    def before_train_epoch(self, runner):
        """检查训练数据集是否与头部兼容。

        参数：
            runner (obj:`EpochBasedRunner`): 基于周期的运行器。
        """
        self._check_head(runner, runner.data_loader.dataset)

    def before_val_epoch(self, runner):
        """检查评估数据集是否与头部兼容。

        参数：
            runner (obj:`EpochBasedRunner`): 基于周期的运行器。
        """
        self._check_head(runner, runner.data_loader.dataset)
