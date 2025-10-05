"""PyTorch Lamb optimizer w/ behaviour similar to NVIDIA FusedLamb.

This optimizer code was adapted from the following (starting with latest)
* https://github.com/HabanaAI/Model-References/blob/
2b435114fe8e31f159b1d3063b8280ae37af7423/PyTorch/nlp/bert/pretraining/lamb.py
* https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
LanguageModeling/Transformer-XL/pytorch/lamb.py
* https://github.com/cybertronai/pytorch-lamb

Use FusedLamb if you can (GPU). The reason for including this variant of Lamb
is to have a version that is
similar in behaviour to APEX FusedLamb if you aren't using NVIDIA GPUs or
cannot install/use APEX.

In addition to some cleanup, this Lamb impl has been modified to support
PyTorch XLA and has been tested on TPU.

Original copyrights for above sources are below.

Modifications Copyright 2021 Ross Wightman
"""
"""
PyTorch Lamb 优化器，其行为类似于 NVIDIA FusedLamb。

此优化器代码改编自以下内容（从最新开始）：
* https://github.com/HabanaAI/Model-References/blob/
2b435114fe8e31f159b1d3063b8280ae37af7423/PyTorch/nlp/bert/pretraining/lamb.py
* https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
LanguageModeling/Transformer-XL/pytorch/lamb.py
* https://github.com/cybertronai/pytorch-lamb

如果可以的话，请使用 FusedLamb（GPU）。包括此变体的 Lamb 的原因是，如果您没有使用 NVIDIA GPU 或无法安装/使用 APEX，则可以使用与 APEX FusedLamb 行为相似的版本。

除了进行一些清理之外，此 Lamb 实现已被修改以支持 PyTorch XLA，并已在 TPU 上进行测试。

上述来源的原始版权如下。

修改版权 2021 Ross Wightman
"""
# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License
#
# Copyright (c) 2019 cybertronai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math

import torch
from mmcv.runner import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class Lamb(Optimizer):
    """一个纯 PyTorch 变体的 FuseLAMB (NvLamb 变体) 优化器。

    这个类是从 `timm`_ 复制的。LAMB 在 `Large Batch
    Optimization for Deep Learning - Training BERT in 76 minutes`_ 中提出。

    .. _timm:
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/lamb.py
    .. _Large Batch Optimization for Deep Learning - Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962

    参数：
        params (可迭代): 要优化的参数的可迭代对象或定义
        参数组的字典。
        lr (浮点数，可选): 学习率。 (默认: 1e-3)
        betas (元组[浮点数, 浮点数]，可选): 用于计算
            梯度及其范数的运行平均值的系数。 (默认: (0.9, 0.999))
        eps (浮点数，可选): 添加到分母中的项以改善
            数值稳定性。 (默认: 1e-8)
        weight_decay (浮点数，可选): 权重衰减 (L2 惩罚) (默认: 0)
        grad_averaging (布尔值，可选): 在计算梯度的运行平均值时
            是否应用 (1-beta2) 到梯度。 (默认: True)
        max_grad_norm (浮点数，可选): 用于剪切全局梯度范数的值
            (默认: 1.0)
        trust_clip (布尔值): 启用 LAMBC 信任比率剪切 (默认: False)
        always_adapt (布尔值，可选): 将自适应学习率应用于 0.0
            权重衰减参数 (默认: False)
    """  # noqa: E501

    def __init__(self,
                 params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-6,
                 weight_decay=0.01,
                 grad_averaging=True,
                 max_grad_norm=1.0,
                 trust_clip=False,
                 always_adapt=False):
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_averaging=grad_averaging,
            max_grad_norm=max_grad_norm,
            trust_clip=trust_clip,
            always_adapt=always_adapt)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行单个优化步骤。

        参数：
            closure（可调用，可选）：一个重新评估模型并返回损失的闭包。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = self.param_groups[0]['params'][0].device
        one_tensor = torch.tensor(
            1.0, device=device
        )  # because torch.where doesn't handle scalars correctly
        global_grad_norm = torch.zeros(1, device=device)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'Lamb does not support sparse gradients, consider '
                        'SparseAdam instead.')
                global_grad_norm.add_(grad.pow(2).sum())

        global_grad_norm = torch.sqrt(global_grad_norm)
        # FIXME 如果在torch.where正确提升标量类型时，去掉标量的显式张量转换会很好
        #  https://github.com/pytorch/pytorch/issues/9190
        max_grad_norm = torch.tensor(
            self.defaults['max_grad_norm'], device=device)
        clip_global_grad_norm = torch.where(global_grad_norm > max_grad_norm,
                                            global_grad_norm / max_grad_norm,
                                            one_tensor)

        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0
            beta3 = 1 - beta1 if grad_averaging else 1.0

            # 假设现在在组内采用相同的步骤以简化事情
            # 每个参数的步骤可以通过将其转换为张量轻松支持，或者
            # 将列表传递到内核中
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            if bias_correction:
                bias_correction1 = 1 - beta1**group['step']
                bias_correction2 = 1 - beta2**group['step']
            else:
                bias_correction1, bias_correction2 = 1.0, 1.0

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.div_(clip_global_grad_norm)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient valuesa
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=beta3)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2)  # v_t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group['eps'])
                update = (exp_avg / bias_correction1).div_(denom)

                weight_decay = group['weight_decay']
                if weight_decay != 0:
                    update.add_(p, alpha=weight_decay)

                if weight_decay != 0 or group['always_adapt']:
                    # 分层学习率适应。默认情况下，跳过对
                    # 排除在权重衰减之外的参数的适应，除非 always_adapt == True，
                    # 否则始终启用。
                    w_norm = p.norm(2.0)
                    g_norm = update.norm(2.0)
                    # FIXME nested where required since logical and/or not
                    #  working in PT XLA
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, w_norm / g_norm, one_tensor),
                        one_tensor,
                    )
                    if group['trust_clip']:
                        # LAMBC trust clipping, upper bound fixed at one
                        trust_ratio = torch.minimum(trust_ratio, one_tensor)
                    update.mul_(trust_ratio)

                p.add_(update, alpha=-group['lr'])

        return loss
