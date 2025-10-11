import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse # 处理命令行参数和选项
import datetime
import random
import time
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm
from mmengine.optim.scheduler.lr_scheduler import PolyLR
import swanlab

import util.misc as utils
from engine import train_one_epoch
from models import build_model
from dataset import create_dataset
from eval.evaluate import cal_mIoU_metrics, cal_ODS_metrics, cal_OIS_metrics
from util.logger import get_logger
from util.early_stopping import EarlyStopping
from util.notify import send_training_completion_notification

def get_args_parser():
    """
    设定默认参数，也可以命令行输入修改
    """
    parser = argparse.ArgumentParser('SEFG', add_help=False)

    # --- 损失 / 模型相关 ---
    loss_group = parser.add_argument_group('Loss / Model')
    loss_group.add_argument('--bce_weight', default=3.0, type=float, help='二元交叉熵损失(bce)的权重。')
    loss_group.add_argument('--iou_weight', default=1.0, type=float, help='IoU损失的权重。')
    loss_group.add_argument('--Norm_Type', default='GN', type=str, help='归一化层类型 [GN|BN], GN=组归一化 ')

    # --- 模型评价指标 ---
    metric_eval = parser.add_argument_group('Metrics')
    metric_eval.add_argument('--eval_metric1', choices=['mIoU', 'ODS_F1'], default='mIoU', type=str, help='更新模型参考的评价指标1')
    metric_eval.add_argument('--eval_metric2', choices=['mIoU', 'OIS_F1'], default='mIoU', type=str, help='更新模型参考的评价指标2')

    # --- 数据集 / 输入 ---
    dataset_group = parser.add_argument_group('Dataset / Input')
    # 用于autodl
    # dataset_group.add_argument('--dataset_path', default="/root/autodl-tmp/data/CRACK500", help='数据集的根目录')
    # 用于本地      
    dataset_group.add_argument('--dataset_path', default="data/DeepCrack", help='数据集的目录')
    dataset_group.add_argument('--dataset_mode', type=str, default='crack', help='数据集类型')
    dataset_group.add_argument('--load_width', type=int, default=256, help='输入图像的宽度以进行预处理（将被调整大小）')
    dataset_group.add_argument('--load_height', type=int, default=256, help='输入图像的高度以进行预处理（将被调整大小）')

    # --- 数据处理 / 批次 ---
    dataloader_group = parser.add_argument_group('DataLoader / Batch')
    dataloader_group.add_argument('--batch_size_train', type=int, default=1, help='每个训练批次的样本数量（影响内存使用）')
    dataloader_group.add_argument('--batch_size_test', type=int, default=1, help='每批样本数量')
    dataloader_group.add_argument('--num_threads', default=0, type=int, help='数据加载的子进程数量')
    dataloader_group.add_argument('--serial_batches', action='store_true', help='禁用随机洗牌，如果启用则使用顺序批量采样。')

    # --- 输出 ---
    output_group = parser.add_argument_group('Output / IO')
    # 用于autodl
    # output_group.add_argument('--output_dir', default='/root/autodl-tmp/output/SFEG-re/checkpoints', help='保存模型参数的输出目录')
    # output_group.add_argument('--output_log', default='/root/autodl-tmp/output/SFEG-re/logs', help='保存训练日志的输出目录')
    # output_group.add_argument('--output_results', default='/root/autodl-tmp/output/SFEG-re/results', help='保存测试结果的输出目录')
    # 用于本地
    output_group.add_argument('--output_dir', default='checkpoints', help='保存模型参数的输出目录')
    output_group.add_argument('--output_log', default='logs', help='保存训练日志的输出目录')
    output_group.add_argument('--output_results', default='results', help='保存测试结果的输出目录')

    # --- 设备 ---
    device_group = parser.add_argument_group('Device')
    device_group.add_argument('--device', default='cuda', help='训练/推理使用的计算设备')

    # --- 训练相关 ---
    training_group = parser.add_argument_group('Training / Runtime')
    training_group.add_argument('--phase', type=str, default='train', help='运行时阶段选择器')
    training_group.add_argument('--epochs', default=100, type=int, help='训练总批次')
    training_group.add_argument('--start_epoch', default=0, type=int, help='手动开始训练的轮次编号（对恢复训练有用）')
    training_group.add_argument('--seed', default=42, type=int, help='随机种子')

    # --- 优化器 / 学习率 / 正则化 ---
    optim_group = parser.add_argument_group('Optimizer / LR')
    # optim_group.add_argument('--sgd', action='store_true', help='使用SGD优化器替代默认的 Adamw优化器')
    optim_group.add_argument('--optimizer', choices=['adamw', 'adam', 'sgd'], default='adam', help='选择优化器[adamw|adam|sgd]')
    optim_group.add_argument('--lr', default=0.001, type=float, help='初始学习率')
    optim_group.add_argument('--weight_decay', default=1e-4, type=float, help='正则化的权重衰减系数')
    optim_group.add_argument('--lr_scheduler', type=str, default='CosLR', help='学习率调度器类型 [PolyLR|StepLR|CosLR]')
    optim_group.add_argument('--min_lr', default=1e-6, type=float, help='PolyLR的最小学习率')
    optim_group.add_argument('--lr_drop', default=30, type=int, help='学习率在 StepLR 调度器中下降的周期间隔')

    # --- 早停 ---
    early_stop_group = parser.add_argument_group('Early Stopping')
    early_stop_group.add_argument('--disable-early-stopping', action='store_true', help='禁用早停机制（默认开启）')
    early_stop_group.add_argument('--patience', default=20, type=int, help='早停耐心值：连续多少个epoch无改善则停止训练')
    early_stop_group.add_argument('--min_delta', default=0.0005, type=float, help='指标改善的最小阈值')
    early_stop_group.add_argument('--lr_patience', default=10, type=int, help='学习率衰减的耐心值 (应小于或等于patience)')
    early_stop_group.add_argument('--lr_factor', default=0.5, type=float, help='学习率衰减因子')
    early_stop_group.add_argument('--restore_best_weights', action='store_true', help='早停时恢复到最佳权重')

    return parser

def evaluate_online(model, data_loader, device, epoch, save_path=None):
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
    model.eval()
    all_pred_probs = []
    all_gt_maps = []
    all_filenames = []
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"Validating Epoch {epoch}"):
            x = data["image"].to(device)
            target = data["label"].to(device)
            target[target > 0] = 1
            out, _, _ = model(x)
            pred_probs = torch.sigmoid(out)
            all_pred_probs.extend([p.cpu() for p in pred_probs])
            all_gt_maps.extend([g.cpu() for g in target])
            if "A_paths" in data:
                all_filenames.extend([Path(p).stem for p in data["A_paths"]])
    if not all_pred_probs:
        print("警告：评估数据为空，跳过指标计算和图片保存。")
        return {'epoch': epoch}
    # 转换成评估函数期望的Numpy列表格式
    # 每个元素都是一个(H, W)的数组
    pred_numpy_list = [(p[0].numpy() * 255).astype(np.uint8) for p in all_pred_probs]
    gt_numpy_list = [(g[0].numpy() * 255).astype(np.uint8) for g in all_gt_maps]
    ods_f1, ods_p, ods_r, ods_threshold = cal_ODS_metrics(pred_numpy_list, gt_numpy_list)
    if save_path:
        print(f"\n正在使用全局最佳阈值 {ods_threshold:.4f} 保存预测图至: {save_path}")
        for i, (prob_tensor, gt_tensor) in enumerate(tqdm(zip(all_pred_probs, all_gt_maps), total=len(all_pred_probs), desc="Saving images")):
            # 获取原始文件名（不含扩展名）
            root_name = all_filenames[i] if i < len(all_filenames) else f"image_{i:04d}"
            
            # 将[0, 1]的概率图转换为[0, 255]的灰度图
            prob_map_uint8 = (prob_tensor[0].numpy() * 255).astype(np.uint8)

            # --- 保存预测二值图 ---
            # 使用 ods_threshold 进行二值化
            o_img = (prob_map_uint8 > (ods_threshold * 255)).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(save_path, f"{root_name}_pre.png"), o_img)

            # --- 保存真值图标注 ---
            t_img = (gt_tensor[0].numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_path, f"{root_name}_lab.png"), t_img)
            
            # --- 保存概率图 ---
            cv2.imwrite(os.path.join(save_path, f"{root_name}_prob.png"), prob_map_uint8)

    mIoU = cal_mIoU_metrics(pred_numpy_list, gt_numpy_list)
    ois_f1, ois_p, ois_r = cal_OIS_metrics(pred_numpy_list, gt_numpy_list)
    
    return {
        'epoch': epoch, 'mIoU': mIoU,
        'ODS_F1': ods_f1, 'ODS_P': ods_p, 'ODS_R': ods_r,
        'OIS_F1': ois_f1, 'OIS_P': ois_p, 'OIS_R': ois_r,
        'ODS_Threshold': ods_threshold
    }


def main(args):
    # 使用swanlab追踪训练
    run = swanlab.init(
        project="SFEG-re", 
        experiment_name=f"{os.path.basename(args.dataset_path)}-{args.optimizer}-{args.load_height}x{args.load_width}",
        description="",
        workspace="crack_segmentation",
        config={
            'bce_weight': args.bce_weight,
            'iou_weight': args.iou_weight,
            'load_height': args.load_height,
            'load_width': args.load_width,
            'optimizer': args.optimizer,
            'epochs': args.epochs,
            'lr_scheduler': args.lr_scheduler,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'eval_metric1': args.eval_metric1,
            'eval_metric2': args.eval_metric2
        }
    )

    logs_path = args.output_log
    cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
    dataset_name = (args.dataset_path).split('/')[-1]
    # 存放训练，测试，评价日志的地址
    process_logs_path = os.path.join(logs_path, cur_time + '_Dataset->' + dataset_name)

    if not os.path.exists(process_logs_path):
        os.makedirs(process_logs_path)
    else:
        print("创建存放日志文件夹错误！")
    
    # 训练，测试，评价日志
    log_train = get_logger(process_logs_path, 'train')
    log_test = get_logger(process_logs_path, 'test')
    log_val = get_logger(process_logs_path, 'val')

    # 设置训练设备和随机种子
    device = torch.device(args.device)
    # 后面的是为了分布式训练时每个进程的种子有差异，单进程时后面函数返回0
    seed = args.seed + utils.get_rank() 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 是否启用早停
    early_stopper = None
    monitor_name = f'{args.eval_metric1}+{args.eval_metric2}'
    if not args.disable_early_stopping:
        early_stopper = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            monitor=monitor_name, # 使用复合指标名称
            lr_patience=args.lr_patience,
            lr_factor=args.lr_factor,
            restore_best_weights=args.restore_best_weights,
            mode='max'  # ODS, OIS, mIoU都是越大越好
        )
        log_train.info(f"早停机制已启用: 监控 '{monitor_name}', 耐心值={args.patience}")
        print(f"早停机制已启用: 监控 '{monitor_name}', 耐心值={args.patience}")
    
    # 创建数据集
    # 训练集
    args.phase = 'train'
    args.batch_size = args.batch_size_train
    train_dataLoader = create_dataset(args)
    print(f'训练数据集大小 = {len(train_dataLoader)}')
    log_train.info(f'训练数据集大小 = {len(train_dataLoader)}')
    # 验证集
    args.phase = 'val'
    args.batch_size = args.batch_size_test
    val_dataLoader = create_dataset(args)
    print(f'验证数据集大小 = {len(val_dataLoader)}')
    log_train.info(f'验证数据集大小 = {len(val_dataLoader)}')

    # 创建模型
    model, criterion = build_model(args)
    model.to(device)

    # 设置优化器
    if args.optimizer == 'sgd':
        print('使用SGD优化器!')
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        print('使用Adam优化器!')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        print('使用AdamW优化器!')
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    
    # 设置学习率调度器
    if args.lr_scheduler == 'StepLR': # 阶梯式衰减
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'CosLR': # 余弦退火
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif args.lr_scheduler == 'PolyLR': # 多项式衰减
        lr_scheduler = PolyLR(optimizer, eta_min=args.min_lr, begin=args.start_epoch, end=args.epochs)
    else:
        raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")
    
    # 存放模型参数的地址
    # 如./checkpoints/2025_10_06_12:00:00_Dataset->TUT"
    output_dir = Path(args.output_dir) / f"{cur_time}_Dataset->{dataset_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("开始训练！")
    log_train.info("开始训练！")
    start_time = time.time()
    # 用于记录综合评价指标（可根据 --eval_metric1/--eval_metric2 选择）
    max_Metrics = {'epoch': -1, 'mIoU': 0, 'ODS_F1': 0, 'ODS_P': 0, 'ODS_R': 0, 'OIS_F1': 0, 'OIS_P': 0, 'OIS_R': 0}

    for epoch in range(args.start_epoch, args.epochs):
        print("---------------------------------------------------------------------------------------")
        print("开始训练 epoch -> ", epoch)
        # 训练一个 epoch
        args.phase = 'train'
        SLoss = train_one_epoch(model, criterion, train_dataLoader, optimizer, epoch, args, log_train)
        log_train.info(f"Epoch -> {epoch} | SLoss -> {SLoss} | lr -> {optimizer.param_groups[0]['lr']}")
        # 使用早停机制更新学习率
        if not (early_stopper and early_stopper.lr_patience < early_stopper.patience):
            lr_scheduler.step()
        print("结束训练 epoch -> ", epoch)
        print("---------------------------------------------------------------------------------------")
        print("开始验证 epoch -> ", epoch)
        val_metrics = evaluate_online(model, val_dataLoader, device, epoch)

        for key, value in val_metrics.items():
            if isinstance(value, float):
                print(f'Validation {key} -> {value:.4f}')
                log_val.info(f"Epoch {epoch} | Validation {key} -> {value:.4f}")
            swanlab.log({f'val_{key}': value, 'epoch': epoch})
        print(f"结束验证 epoch -> {epoch}")
        print("---------------------------------------------------------------------------------------")
        # 使用早停或常规逻辑保存最佳模型
        eval_val = val_metrics.get(args.eval_metric1, 0) + val_metrics.get(args.eval_metric2, 0)
        if early_stopper:
            should_stop = early_stopper(
                current_score=eval_val, epoch=epoch, model=model, optimizer=optimizer,
                lr_scheduler=lr_scheduler, output_dir=output_dir, logger=log_train,
                extra_save_info={'args': args}
            )
            if early_stopper.best_epoch == epoch:
                max_Metrics = val_metrics
            if should_stop:
                print("早停条件已满足，训练提前结束。")
                log_train.info("早停条件已满足，训练提前结束。")
                break
        else: # 无早停时的逻辑
            current_max_val = max_Metrics.get(args.eval_metric1, 0) + max_Metrics.get(args.eval_metric2, 0)
            if eval_val > current_max_val:
                max_Metrics = val_metrics
                checkpoint_path = output_dir / 'checkpoint_best.pth'
                utils.save_on_master({
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch, 'args': args,
                }, checkpoint_path)
                log_train.info(f"\n更新并保存最佳模型 -> {epoch}")
                print(f"\n更新并保存最佳模型 -> {epoch}")

    print("---------------------------------------------------------------------------------------")

    print("开始最终测试... 加载最佳模型...")
    log_test.info("开始最终测试... 加载最佳模型...")
    final_test_metrics = {}
    best_model_path = output_dir / 'checkpoint_best.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        best_epoch = checkpoint['epoch']
        print(f"已加载 Epoch {best_epoch} 的最佳模型。")
        log_test.info(f"已加载 Epoch {best_epoch} 的最佳模型。")
        final_results_path = Path(args.output_results) / f"{cur_time}_Dataset->{dataset_name}"
        
        # 创建测试集
        args.phase = 'test'
        args.batch_size = args.batch_size_test
        test_dataLoader = create_dataset(args)
        print(f"测试数据集大小 = {len(test_dataLoader)}")
        log_test.info(f"测试数据集大小 = {len(test_dataLoader)}")
        # 在测试集上进行最终评估
        final_test_metrics = evaluate_online(
            model=model, 
            data_loader=test_dataLoader, 
            device=device, 
            epoch=best_epoch,
            save_path=final_results_path
        )
        
        print("\n--- 最终模型在测试集上的性能 ---")
        log_test.info("\n--- 最终模型在测试集上的性能 ---")
        for key, value in final_test_metrics.items():
            if isinstance(value, float):
                print(f'Test {key} -> {value:.4f}')
                log_test.info(f'Test {key} -> {value:.4f}')
    else:
        print("未找到最佳模型文件，跳过最终测试。")
        log_test.warning("未找到最佳模型文件，跳过最终测试。")
    print("---------------------------------------------------------------------------------------")

    # Autodl 训练完成通知（使用最终测试集得分）
    # send_training_completion_notification(
    #     final_metrics=final_test_metrics,
    #     dataset_name=dataset_name,
    #     early_stopper=early_stopper,
    #     total_epochs=args.epochs,
    #     eval_metric1=args.eval_metric1,
    #     eval_metric2=args.eval_metric2
    # )

    # 把最佳评价得分写入日志
    for key, value in max_Metrics.items():
        if isinstance(value, float):
            print(f'Best Validation {key} -> {value:.4f}')
            log_val.info(f'Best Validation {key} -> {value:.4f}')
        else:
            print(f'Best Validation Epoch -> {value}')
            log_val.info(f'Best Validation Epoch -> {value}')

    print("---------------------------------------------------------------------------------------")
    # 打印早停总结
    if early_stopper:
        summary = early_stopper.get_summary()
        summary_str = (
            f"\n{'='*20} 早停总结 {'='*20}\n"
            f"监控指标: {summary['monitor_metric']}\n"
            f"最佳分数: {summary['best_score']:.6f}\n"
            f"最佳Epoch: {summary['best_epoch']}\n"
            f"停止Epoch: {summary['stopped_epoch']}\n"
            f"学习率降低次数: {summary['lr_reductions']}\n"
            f"{'='*50}"
        )
        print(summary_str)
        log_train.info(summary_str)
        print("---------------------------------------------------------------------------------------")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\n总训练时间为 {total_time_str}')
    log_train.info(f'总训练时间为 {total_time_str}')
    
    # 将训练配置参数记录到日志里
    log_train.info('--- 最终使用的训练配置 ---')
    log_train.info(f'args -> {str(args)}')
    log_train.info(f'bce_weight -> {args.bce_weight}')
    log_train.info(f'iou_weight -> {args.iou_weight}')
    log_train.info(f'数据集 -> {args.dataset_path}')
    log_train.info(f'输入图像尺寸 -> {args.load_width}x{args.load_height}')
    log_train.info(f'优化器 -> {args.optimizer}')
    log_train.info(f'训练总批次 -> {args.epochs}')
    log_train.info(f'学习率更新策略 -> {args.lr_scheduler}')
    log_train.info(f'初始学习率 -> {args.lr}')
    log_train.info(f'权重衰减 -> {args.weight_decay}')
    log_train.info(f'模型更新评价指标1 -> {args.eval_metric1}')
    log_train.info(f'模型更新评价指标2 -> {args.eval_metric2}')

    print('训练时间为 {}'.format(total_time_str))
    log_train.info('训练时间为 {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SFEG', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
