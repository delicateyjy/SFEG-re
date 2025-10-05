# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
from mmcv.runner import HOOKS, BaseRunner
from mmcv.runner.dist_utils import get_dist_info, master_only
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.hooks.evaluation import DistEvalHook, EvalHook
from mmcv.runner.hooks.logger.wandb import WandbLoggerHook


@HOOKS.register_module()
class MMClsWandbHook(WandbLoggerHook):
    """增强的Wandb日志记录器钩子用于分类。

    与:cls:`mmcv.runner.WandbLoggerHook`相比，这个钩子不仅可以自动记录``log_buffer``中的所有信息，还可以记录以下额外信息。

    - **检查点**：如果``log_checkpoint``为True，则在每个检查点间隔保存的检查点将作为W&B工件保存。这依赖于:class:`mmcv.runner.CheckpointHook`，其优先级高于此钩子。请参阅https://docs.wandb.ai/guides/artifacts/model-versioning以了解有关使用W&B工件进行模型版本控制的更多信息。

    - **检查点元数据**：如果``log_checkpoint_metadata``为True，则每个检查点工件将具有相关联的元数据。元数据包含在验证数据上计算的与该检查点相关的评估指标以及当前的epoch/iter。它依赖于优先级高于此钩子的:class:`EvalHook`。

    - **评估**：在每个间隔，此钩子将模型预测记录为交互式W&B表。记录的样本数量由``num_eval_images``给出。目前，此钩子在每个评估间隔记录预测标签和真实标签。这依赖于优先级高于此钩子的:class:`EvalHook`。还要注意，数据仅记录一次，后续评估表使用对已记录数据的引用以节省内存使用。请参阅https://docs.wandb.ai/guides/data-vis以了解有关W&B表的更多信息。

    这里是一个配置示例：

    .. code:: python

        checkpoint_config = dict(interval=10)

        # 要记录检查点元数据，检查点保存的间隔应该
        # 能被评估的间隔整除。
        evaluation = dict(interval=5)

        log_config = dict(
            ...
            hooks=[
                ...
                dict(type='MMClsWandbHook',
                     init_kwargs={
                         'entity': "YOUR_ENTITY",
                         'project': "YOUR_PROJECT_NAME"
                     },
                     log_checkpoint=True,
                     log_checkpoint_metadata=True,
                     num_eval_images=100)
            ])

    参数：
        init_kwargs (dict)：传递给wandb.init以初始化
            W&B运行的字典。请参阅https://docs.wandb.ai/ref/python/init
            以获取可能的键值对。
        interval (int)：记录间隔（每k次迭代）。默认为10。
        log_checkpoint (bool)：在每个检查点间隔
            将检查点保存为W&B工件。将其用于模型版本控制，其中每个版本
            是一个检查点。默认为False。
        log_checkpoint_metadata (bool)：记录在验证数据上计算的评估指标
            与检查点的当前epoch一起作为该检查点的元数据。
            默认为True。
        num_eval_images (int)：要记录的验证图像数量。
            如果为零，则评估将不会被记录。默认为100。
    """

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 log_checkpoint=False,
                 log_checkpoint_metadata=False,
                 num_eval_images=100,
                 **kwargs):
        super(MMClsWandbHook, self).__init__(init_kwargs, interval, **kwargs)

        self.log_checkpoint = log_checkpoint
        self.log_checkpoint_metadata = (
            log_checkpoint and log_checkpoint_metadata)
        self.num_eval_images = num_eval_images
        self.log_evaluation = (num_eval_images > 0)
        self.ckpt_hook: CheckpointHook = None
        self.eval_hook: EvalHook = None

    @master_only
    def before_run(self, runner: BaseRunner):
        super(MMClsWandbHook, self).before_run(runner)

        # Inspect CheckpointHook and EvalHook
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook
            if isinstance(hook, (EvalHook, DistEvalHook)):
                self.eval_hook = hook

        # Check conditions to log checkpoint
        if self.log_checkpoint:
            if self.ckpt_hook is None:
                self.log_checkpoint = False
                self.log_checkpoint_metadata = False
                runner.logger.warning(
                    '在 MMClsWandbHook 中记录检查点，需要 `CheckpointHook`，请检查运行器中的钩子。')
            else:
                self.ckpt_interval = self.ckpt_hook.interval

        # Check conditions to log evaluation
        if self.log_evaluation or self.log_checkpoint_metadata:
            if self.eval_hook is None:
                self.log_evaluation = False
                self.log_checkpoint_metadata = False
                runner.logger.warning(
                    '要在`MMClsWandbHook`中记录评估或检查点元数据，mmcls中的`EvalHook`或`DistEvalHook`是必需的，请检查验证是否已启用。')
            else:
                self.eval_interval = self.eval_hook.interval
            self.val_dataset = self.eval_hook.dataloader.dataset
            if (self.log_evaluation
                    and self.num_eval_images > len(self.val_dataset)):
                self.num_eval_images = len(self.val_dataset)
                runner.logger.warning(
                    f'The num_eval_images ({self.num_eval_images}) is '
                    'greater than the total number of validation samples '
                    f'({len(self.val_dataset)}). The complete validation '
                    'dataset will be logged.')

        # Check conditions to log checkpoint metadata
        if self.log_checkpoint_metadata:
            assert self.ckpt_interval % self.eval_interval == 0, \
                'To log checkpoint metadata in MMClsWandbHook, the interval ' \
                f'of checkpoint saving ({self.ckpt_interval}) should be ' \
                'divisible by the interval of evaluation ' \
                f'({self.eval_interval}).'

        # Initialize evaluation table
        if self.log_evaluation:
            # Initialize data table
            self._init_data_table()
            # Add ground truth to the data table
            self._add_ground_truth()
            # Log ground truth data
            self._log_data_table()

    @master_only
    def after_train_epoch(self, runner):
        super(MMClsWandbHook, self).after_train_epoch(runner)

        if not self.by_epoch:
            return

        # Save checkpoint and metadata
        if (self.log_checkpoint
                and self.every_n_epochs(runner, self.ckpt_interval)
                or (self.ckpt_hook.save_last and self.is_last_epoch(runner))):
            if self.log_checkpoint_metadata and self.eval_hook:
                metadata = {
                    'epoch': runner.epoch + 1,
                    **self._get_eval_results()
                }
            else:
                metadata = None
            aliases = [f'epoch_{runner.epoch+1}', 'latest']
            model_path = osp.join(self.ckpt_hook.out_dir,
                                  f'epoch_{runner.epoch+1}.pth')
            self._log_ckpt_as_artifact(model_path, aliases, metadata)

        # Save prediction table
        if self.log_evaluation and self.eval_hook._should_evaluate(runner):
            results = self.eval_hook.latest_results
            # Initialize evaluation table
            self._init_pred_table()
            # Add predictions to evaluation table
            self._add_predictions(results, runner.epoch + 1)
            # Log the evaluation table
            self._log_eval_table(runner.epoch + 1)

    def after_train_iter(self, runner):
        if self.get_mode(runner) == 'train':
            # 一个丑陋的补丁。基于迭代的评估钩子将在评估之前调用所有记录器钩子的
            # `after_train_iter` 方法。
            # 使用这个技巧来跳过该调用。
            # 不要先调用父类方法，这会清除 log_buffer
            return super(MMClsWandbHook, self).after_train_iter(runner)
        else:
            super(MMClsWandbHook, self).after_train_iter(runner)

        rank, _ = get_dist_info()
        if rank != 0:
            return

        if self.by_epoch:
            return

        # Save checkpoint and metadata
        if (self.log_checkpoint
                and self.every_n_iters(runner, self.ckpt_interval)
                or (self.ckpt_hook.save_last and self.is_last_iter(runner))):
            if self.log_checkpoint_metadata and self.eval_hook:
                metadata = {
                    'iter': runner.iter + 1,
                    **self._get_eval_results()
                }
            else:
                metadata = None
            aliases = [f'iter_{runner.iter+1}', 'latest']
            model_path = osp.join(self.ckpt_hook.out_dir,
                                  f'iter_{runner.iter+1}.pth')
            self._log_ckpt_as_artifact(model_path, aliases, metadata)

        # Save prediction table
        if self.log_evaluation and self.eval_hook._should_evaluate(runner):
            results = self.eval_hook.latest_results
            # Initialize evaluation table
            self._init_pred_table()
            # Log predictions
            self._add_predictions(results, runner.iter + 1)
            # Log the table
            self._log_eval_table(runner.iter + 1)

    @master_only
    def after_run(self, runner):
        self.wandb.finish()

    def _log_ckpt_as_artifact(self, model_path, aliases, metadata=None):
        """记录模型检查点作为 W&B 工件。

        参数：
            model_path (str)：要记录的检查点路径。
            aliases (list)：与此工件相关的别名列表。
            metadata (dict，可选)：与此工件相关的元数据。
        """
        model_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_model', type='model', metadata=metadata)
        model_artifact.add_file(model_path)
        self.wandb.log_artifact(model_artifact, aliases=aliases)

    def _get_eval_results(self):
        """Get model evaluation results."""
        results = self.eval_hook.latest_results
        eval_results = self.val_dataset.evaluate(
            results, logger='silent', **self.eval_hook.eval_kwargs)
        return eval_results

    def _init_data_table(self):
        """Initialize the W&B Tables for validation data."""
        columns = ['image_name', 'image', 'ground_truth']
        self.data_table = self.wandb.Table(columns=columns)

    def _init_pred_table(self):
        """Initialize the W&B Tables for model evaluation."""
        columns = ['epoch'] if self.by_epoch else ['iter']
        columns += ['image_name', 'image', 'ground_truth', 'prediction'
                    ] + list(self.val_dataset.CLASSES)
        self.eval_table = self.wandb.Table(columns=columns)

    def _add_ground_truth(self):
        # Get image loading pipeline
        from mmcls.datasets.pipelines import LoadImageFromFile
        img_loader = None
        for t in self.val_dataset.pipeline.transforms:
            if isinstance(t, LoadImageFromFile):
                img_loader = t

        CLASSES = self.val_dataset.CLASSES
        self.eval_image_indexs = np.arange(len(self.val_dataset))
        # Set seed so that same validation set is logged each time.
        np.random.seed(42)
        np.random.shuffle(self.eval_image_indexs)
        self.eval_image_indexs = self.eval_image_indexs[:self.num_eval_images]

        for idx in self.eval_image_indexs:
            img_info = self.val_dataset.data_infos[idx]
            if img_loader is not None:
                img_info = img_loader(img_info)
                # Get image and convert from BGR to RGB
                image = img_info['img'][..., ::-1]
            else:
                # For CIFAR dataset.
                image = img_info['img']
            image_name = img_info.get('filename', f'img_{idx}')
            gt_label = img_info.get('gt_label').item()

            self.data_table.add_data(image_name, self.wandb.Image(image),
                                     CLASSES[gt_label])

    def _add_predictions(self, results, idx):
        table_idxs = self.data_table_ref.get_index()
        assert len(table_idxs) == len(self.eval_image_indexs)

        for ndx, eval_image_index in enumerate(self.eval_image_indexs):
            result = results[eval_image_index]

            self.eval_table.add_data(
                idx, self.data_table_ref.data[ndx][0],
                self.data_table_ref.data[ndx][1],
                self.data_table_ref.data[ndx][2],
                self.val_dataset.CLASSES[np.argmax(result)], *tuple(result))

    def _log_data_table(self):
        """记录验证数据的W&B表作为工件，并对其调用`use_artifact`，以便评估表可以使用已上传图像的引用。

        这允许数据只上传一次。
        """
        data_artifact = self.wandb.Artifact('val', type='dataset')
        data_artifact.add(self.data_table, 'val_data')

        self.wandb.run.use_artifact(data_artifact)
        data_artifact.wait()

        self.data_table_ref = data_artifact.get('val_data')

    def _log_eval_table(self, idx):
        """记录 W&B 表格以进行模型评估。

        表格将被多次记录以创建新版本。使用此功能可以交互式地比较不同时间间隔的模型。
        """
        pred_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_pred', type='evaluation')
        pred_artifact.add(self.eval_table, 'eval_data')
        if self.by_epoch:
            aliases = ['latest', f'epoch_{idx}']
        else:
            aliases = ['latest', f'iter_{idx}']
        self.wandb.run.log_artifact(pred_artifact, aliases=aliases)
