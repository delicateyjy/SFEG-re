# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmcls.datasets.pipelines import Compose
from mmcls.models import build_classifier


def init_model(config, checkpoint=None, device='cuda:0', options=None):
    """从配置文件初始化分类器。

    参数：
        config（字符串或 :obj:`mmcv.Config`）：配置文件路径或配置对象。
        checkpoint（字符串，可选）：检查点路径。如果保持为None，模型将不加载任何权重。
        options（字典）：用来覆盖所用配置中某些设置的选项。

    返回值：
        nn.Module：构建的分类器。
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if options is not None:
        config.merge_from_dict(options)
    config.model.pretrained = None
    model = build_classifier(config.model)
    if checkpoint is not None:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            from mmcls.datasets import ImageNet
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use imagenet by default.')
            model.CLASSES = ImageNet.CLASSES
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_model(model, img):
    """用分类器推断图像。

    参数：
        model (nn.Module): 已加载的分类器。
        img (str/ndarray): 图像文件名或已加载的图像。

    返回：
        result (dict): 分类结果，包括
            `class_name`（类别名称），`pred_label`（预测标签）和`pred_score`（预测分数）。
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)
        pred_score = np.max(scores, axis=1)[0]
        pred_label = np.argmax(scores, axis=1)[0]
        result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
    result['pred_class'] = model.CLASSES[result['pred_label']]
    return result


def show_result_pyplot(model,
                       img,
                       result,
                       fig_size=(15, 10),
                       title='result',
                       wait_time=0):
    """在图像上可视化分类结果。

    参数：
        model (nn.Module)：已加载的分类器。
        img (str 或 np.ndarray)：图像文件名或加载的图像。
        result (list)：分类结果。
        fig_size (tuple)：pyplot 图形的大小。
            默认为 (15, 10)。
        title (str)：pyplot 图形的标题。
            默认为 'result'。
        wait_time (int)：显示图像的秒数。
            默认为 0。
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        show=True,
        fig_size=fig_size,
        win_name=title,
        wait_time=wait_time)
