"""
早停机制模块 (Early Stopping Module)
提供一个稳健的早停管理器类，支持多指标监控、学习率衰减和模型权重恢复等功能。
该模块与训练循环解耦，以提高代码的可读性和可维护性。
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union

from . import misc as utils 

class EarlyStopping:
    """
    稳健的早停机制实现

    该类监控一个由主程序传入的单一性能指标值（可以是复合指标，如 ODS+OIS）。
    当该指标在连续多个epoch内不再改善时，可以自动调整学习率或提前终止训练。

    核心功能:
        - *通用指标监控*: 支持 'max' 和 'min' 模式，监控任何数值型指标。
        - *智能学习率调度*: 在性能停滞时自动降低学习率。
        - *最佳权重管理*: 自动保存最佳模型 (`checkpoint_best.pth`)，并支持早停时恢复。
    """
    def __init__(self,
                 patience: int = 15,
                 min_delta: float = 0.001,
                 monitor: str = 'mIoU',  # monitor 现在主要用于日志记录
                 lr_patience: int = 8,
                 lr_factor: float = 0.5,
                 restore_best_weights: bool = True,
                 mode: str = 'max',
                 max_lr_reductions: int = 3,
                 verbose: bool = True):
        
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.restore_best_weights = restore_best_weights
        self.mode = mode.lower()
        self.max_lr_reductions = max_lr_reductions
        self.verbose = verbose

        if self.mode not in ['max', 'min']:
            raise ValueError(f"Mode must be 'max' or 'min', but got '{mode}'")

        self.best_score = -np.inf if self.mode == 'max' else np.inf
        self.best_epoch = -1
        self.wait = 0
        self.lr_wait = 0
        self.lr_reduced_count = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.history = []

    def __call__(self,
                 current_score: float,
                 epoch: int,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: Optional[Any] = None,
                 output_dir: Optional[Union[str, Path]] = None,
                 logger: Optional[Any] = None,
                 extra_save_info: Optional[Dict] = None) -> bool:
        """
        执行早停检查。返回一个布尔值，指示是否应该停止训练。
        """
        improved = self._is_improved(current_score)

        if improved:
            is_new_best = epoch != self.best_epoch
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
            self.lr_wait = 0

            # 仅在找到新的最佳epoch时，才保存权重，避免重复操作
            if is_new_best:
                if self.restore_best_weights:
                    self.best_weights = {key: value.cpu().clone() for key, value in model.state_dict().items()}
                if output_dir:
                    self._save_best_model(model, optimizer, lr_scheduler, epoch, output_dir, extra_save_info)
                self._log(f"新的最佳 {self.monitor}: {current_score:.6f} (epoch {epoch})。模型已保存。", logger)
            
            return False
        else:
            self.wait += 1
            self.lr_wait += 1
            self._log(f"{self.monitor} 无改善: {current_score:.6f}, 等待 {self.wait}/{self.patience} epochs", logger)
            if self.lr_wait >= self.lr_patience and self.lr_reduced_count < self.max_lr_reductions:
                self._reduce_lr(optimizer, logger)
                self.lr_wait = 0
                self.lr_reduced_count += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self._log(f"早停触发！在 epoch {epoch} 停止训练。", logger)
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    self._log(f"已恢复到 epoch {self.best_epoch} 的最佳模型权重。", logger)
                return True
        return False
        
    def _is_improved(self, score: float) -> bool:
        return (self.mode == 'max' and score > self.best_score + self.min_delta) or \
               (self.mode == 'min' and score < self.best_score - self.min_delta)

    def _reduce_lr(self, optimizer: torch.optim.Optimizer, logger: Optional[Any]):
        old_lr = optimizer.param_groups[0]['lr']
        new_lr = old_lr * self.lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        self._log(f"学习率衰减: {old_lr:.2e} -> {new_lr:.2e} (第 {self.lr_reduced_count} 次)", logger)

    def _save_best_model(self, model, optimizer, lr_scheduler, epoch, output_dir, extra_save_info):
        checkpoint_path = Path(output_dir) / 'checkpoint_best.pth'
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
            'epoch': epoch,
            'best_score': self.best_score,
            'monitor': self.monitor,
            **(extra_save_info or {})
        }
        utils.save_on_master(save_dict, checkpoint_path)
    
    def _log(self, message: str, logger: Optional[Any]):
        if self.verbose: print(message)
        if logger: logger.info(message)
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'monitor_metric': self.monitor,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'stopped_epoch': self.stopped_epoch if self.stopped_epoch > 0 else 'N/A',
            'lr_reductions': self.lr_reduced_count,
        }
