# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(BaseEvalHook):
    """非分布式评估钩子。

    与 MMCV 中的 ``EvalHook`` 相比，此钩子将保存最新的
    评估结果作为其他钩子使用的属性（例如
    `MMClsWandbHook`）。
    """

    def __init__(self, dataloader, **kwargs):
        super(EvalHook, self).__init__(dataloader, **kwargs)
        self.latest_results = None

    def _do_evaluate(self, runner):
        """执行评估并保存ckpt。"""
        results = self.test_fn(runner.model, self.dataloader)
        self.latest_results = results
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)


class DistEvalHook(BaseDistEvalHook):
    """非分布式评估钩子。

    与 MMCV 中的 ``EvalHook`` 相比，这个钩子将保存最新的
    评估结果作为属性供其他钩子使用（如 `MMClsWandbHook`）。
    """

    def __init__(self, dataloader, **kwargs):
        super(DistEvalHook, self).__init__(dataloader, **kwargs)
        self.latest_results = None

    def _do_evaluate(self, runner):
        """执行评估并保存检查点。"""
        """ BatchNorm的缓冲区（running_mean和running_var）
        的同步在pytorch的DDP中不被支持，这可能导致不同rank的模型性能不一致，
        因此我们将rank 0的BatchNorm缓冲区广播到其他rank以避免这种情况。"""
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = self.test_fn(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        self.latest_results = results
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            """ key_score 可能为 `None`，因此需要跳过保存最佳检查点的操作。"""
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)
