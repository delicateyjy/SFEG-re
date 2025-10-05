# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    no_type_check)


import numpy as np
import torch
from torch.optim import Optimizer
import torch.distributed as dist
from mmcv.runner import (DistSamplerSeedHook, Fp16OptimizerHook,
                         build_optimizer, build_runner, get_dist_info)
import mmcv

from mmcls.core import DistEvalHook, DistOptimizerHook, EvalHook
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.utils import (auto_select_device, get_root_logger,
                         wrap_distributed_model, wrap_non_distributed_model)

@no_type_check
def resume_ema_model(runner,
           checkpoint: str,
           resume_optimizer: bool = True,
           map_location: Union[str, Callable] = 'default') -> None:
    if map_location == 'default':
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            checkpoint = runner.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = runner.load_checkpoint(checkpoint)
    else:
        checkpoint = runner.load_checkpoint(
            checkpoint, map_location=map_location)

    runner._epoch = checkpoint['meta']['epoch']
    runner._iter = checkpoint['meta']['iter']
    if runner.meta is None:
        runner.meta = {}
    runner.meta.setdefault('hook_msgs', {})
    # load `last_ckpt`, `best_score`, `best_ckpt`, etc. for hook messages
    # 加载 `last_ckpt`、`best_score`、`best_ckpt` 等用于钩子消息
    runner.meta['hook_msgs'].update(checkpoint['meta'].get('hook_msgs', {}))

    # Re-calculate the number of iterations when resuming
    # 恢复时重新计算迭代次数
    # models with different number of GPUs
    if 'config' in checkpoint['meta']:
        config = mmcv.Config.fromstring(
            checkpoint['meta']['config'], file_format='.py')
        previous_gpu_ids = config.get('gpu_ids', None)
        if previous_gpu_ids and len(previous_gpu_ids) > 0 and len(
                previous_gpu_ids) != runner.world_size:
            runner._iter = int(runner._iter * len(previous_gpu_ids) /
                             runner.world_size)
            runner.logger.info('the iteration number is changed due to '
                             'change of GPU number')

    # resume meta information meta
    runner.meta = checkpoint['meta']

    if 'optimizer' in checkpoint and resume_optimizer:
        if isinstance(runner.optimizer, Optimizer):
            runner.optimizer.load_state_dict(checkpoint['optimizer'])
        elif isinstance(runner.optimizer, dict):
            for k in runner.optimizer.keys():
                runner.optimizer[k].load_state_dict(
                    checkpoint['optimizer'][k])
        else:
            raise TypeError(
                'Optimizer should be dict or torch.optim.Optimizer '
                f'but got {type(runner.optimizer)}')

    runner.logger.info('EMA model resumed epoch %d, iter %d', runner.epoch, runner.iter)




def init_random_seed(seed=None, device=None):
    """初始化随机种子。

    如果未设置种子，系统将自动随机生成种子，
    然后广播给所有进程，以防止潜在的错误。

    参数：
        seed（int，可选）：种子。默认为None。
        device（str）：放置种子的设备。
            默认为'cuda'。

    返回：
        int：使用的种子。
    """
    if seed is not None:
        return seed
    if device is None:
        device = auto_select_device()
    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """设置随机种子。

    参数：
        seed (int)：要使用的种子。
        deterministic (bool)：是否为 CUDNN 后端设置确定性选项，即将 `torch.backends.cudnn.deterministic` 设置为 True，并将 `torch.backends.cudnn.benchmark` 设置为 False。
        默认：False。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                device=None,
                meta=None):
    """训练模型。

    该方法将根据提供的配置构建数据加载器，封装模型并构建一个运行器。

    参数：
        model (:obj:`torch.nn.Module`): 要运行的模型。
        dataset (:obj:`mmcls.datasets.BaseDataset` | List[BaseDataset]):
            用于训练模型的数据集。可以是单个数据集，也可以是与工作流长度相同的数据集列表。
        cfg (:obj:`mmcv.utils.Config`): 实验的配置。
        distributed (bool): 是否在分布式环境中训练模型。默认为 False。
        validate (bool): 是否使用 :obj:`mmcv.runner.EvalHook` 进行验证。默认为 False。
        timestamp (str, optional): 自动生成日志文件名称的时间戳字符串。默认为 None。
        device (str, optional): TODO
        meta (dict, optional): 记录一些重要信息的字典，例如环境信息和种子，这些信息将记录在日志钩子中。默认为 None。
    """
    logger = get_root_logger()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=cfg.ipu_replicas if device == 'ipu' else len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
        seed=cfg.get('seed'),
        sampler_cfg=cfg.get('sampler', None),
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = wrap_distributed_model(
            model,
            cfg.device,
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = wrap_non_distributed_model(
            model, cfg.device, device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    if device == 'ipu':
        if not cfg.runner['type'].startswith('IPU'):
            cfg.runner['type'] = 'IPU' + cfg.runner['type']
        if 'options_cfg' not in cfg.runner:
            cfg.runner['options_cfg'] = {}
        cfg.runner['options_cfg']['replicationFactor'] = cfg.ipu_replicas
        cfg.runner['fp16_cfg'] = cfg.get('fp16', None)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)

    if fp16_cfg is None and device == 'npu':
        fp16_cfg = {'loss_scale': 'dynamic'}

    if fp16_cfg is not None:
        if device == 'ipu':
            from mmcv.device.ipu import IPUFp16OptimizerHook
            optimizer_config = IPUFp16OptimizerHook(
                **cfg.optimizer_config,
                loss_scale=fp16_cfg['loss_scale'],
                distributed=distributed)
        else:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config,
                loss_scale=fp16_cfg['loss_scale'],
                distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))
    if distributed and cfg.runner['type'] == 'EpochBasedRunner':
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # The specific dataloader settings
        val_loader_cfg = {
            **loader_cfg,
            'shuffle': False,  # Not shuffle by default
            'sampler_cfg': None,  # Not use sampler by default
            'drop_last': False,  # Not drop last by default
            **cfg.data.get('val_dataloader', {}),
        }
        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # `EvalHook` needs to be executed after `IterTimerHook`.
        # Otherwise, it will cause a bug if use `IterBasedRunner`.
        # Refers to https://github.com/open-mmlab/mmcv/issues/1261
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
