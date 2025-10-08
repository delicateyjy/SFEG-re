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

import util.misc as utils
from engine import train_one_epoch
from models import build_model
from dataset import create_dataset
from eval.evaluate import eval
from util.logger import get_logger

def get_args_parser():
    """
    设定默认参数，也可以命令行输入修改
    """
    parser = argparse.ArgumentParser('SEFG', add_help=False)

    # --- 损失 / 模型相关 ---
    loss_group = parser.add_argument_group('Loss / Model')
    loss_group.add_argument('--bce_weight', default=3.0, type=float,
                            help='二元交叉熵损失(bce)的权重。')
    loss_group.add_argument('--iou_weight', default=1.0, type=float,
                            help='IoU损失的权重。')
    loss_group.add_argument('--Norm_Type', default='GN', type=str,
                            help='归一化层类型 [GN|BN], GN=组归一化 ')

    # --- 模型评价指标 ---
    metric_eval = parser.add_argument_group('Metrics')
    metric_eval.add_argument('--eval_metric1', choices=['mIoU', 'ODS'], default='ODS', type=str,
                             help='更新模型参考的评价指标1')
    metric_eval.add_argument('--eval_metric2', choices=['mIoU', 'OIS'], default='OIS', type=str,
                             help='更新模型参考的评价指标2')

    # --- 数据集 / 输入 ---
    dataset_group = parser.add_argument_group('Dataset / Input')
    # 用于autodl
    # dataset_group.add_argument('--dataset_path', default="/root/autodl-tmp/data/CRACK500",
    #                            help='数据集的根目录')
    # 用于本地
    dataset_group.add_argument('--dataset_path', default="data/cracktree200",
                               help='数据集的目录')
    dataset_group.add_argument('--dataset_mode', type=str, default='crack',
                               help='数据集类型')
    dataset_group.add_argument('--load_width', type=int, default=256,
                               help='输入图像的宽度以进行预处理（将被调整大小）')
    dataset_group.add_argument('--load_height', type=int, default=256,
                               help='输入图像的高度以进行预处理（将被调整大小）')

    # --- 数据处理 / 批次 ---
    dataloader_group = parser.add_argument_group('DataLoader / Batch')
    dataloader_group.add_argument('--batch_size_train', type=int, default=1,
                                  help='每个训练批次的样本数量（影响内存使用）')
    dataloader_group.add_argument('--batch_size_test', type=int, default=1,
                                  help='每批样本数量')
    dataloader_group.add_argument('--num_threads', default=0, type=int,
                                  help='数据加载的子进程数量')
    dataloader_group.add_argument('--serial_batches', action='store_true',
                                  help='禁用随机洗牌，如果启用则使用顺序批量采样。')

    # --- 输出 ---
    output_group = parser.add_argument_group('Output / IO')
    # 用于autodl
    # output_group.add_argument('--output_dir', default='/root/autodl-tmp/output/SFEG-re/checkpoints',
    #                           help='保存模型参数的输出目录')
    # output_group.add_argument('--output_log', default='/root/autodl-tmp/output/SFEG-re/logs',
    #                           help='保存训练日志的输出目录')
    # output_group.add_argument('--output_results', default='/root/autodl-tmp/output/SFEG-re/results',
    #                           help='保存测试结果的输出目录')
    # 用于本地
    output_group.add_argument('--output_dir', default='checkpoints',
                              help='保存模型参数的输出目录')
    output_group.add_argument('--output_log', default='logs',
                              help='保存训练日志的输出目录')
    output_group.add_argument('--output_results', default='results',
                              help='保存测试结果的输出目录')

    # --- 设备 ---
    device_group = parser.add_argument_group('Device')
    device_group.add_argument('--device', default='cuda',
                              help='训练/推理使用的计算设备')

    # --- 训练相关 ---
    training_group = parser.add_argument_group('Training / Runtime')
    training_group.add_argument('--phase', type=str, default='train',
                                help='运行时阶段选择器')
    training_group.add_argument('--epochs', default=50, type=int,
                                help='训练总批次')
    training_group.add_argument('--start_epoch', default=0, type=int,
                                help='手动开始训练的轮次编号（对恢复训练有用）')
    training_group.add_argument('--seed', default=42, type=int,
                                help='随机种子')

    # --- 优化器 / 学习率 / 正则化 ---
    optim_group = parser.add_argument_group('Optimizer / LR')
    # optim_group.add_argument('--sgd', action='store_true',
    #                          help='使用SGD优化器替代默认的 Adamw优化器')
    optim_group.add_argument('--optimizer', choices=['adamw', 'sgd'], default='sgd',
                             help='选择优化器[adamw|sgd]')
    optim_group.add_argument('--lr_scheduler', type=str, default='PolyLR',
                             help='学习率调度器类型 [PolyLR|StepLR|CosLR]')
    optim_group.add_argument('--lr', default=5e-4, type=float,
                             help='初始学习率')
    optim_group.add_argument('--min_lr', default=1e-6, type=float,
                             help='PolyLR的最小学习率')
    optim_group.add_argument('--lr_drop', default=30, type=int,
                             help='学习率在 StepLR 调度器中下降的周期间隔')
    optim_group.add_argument('--weight_decay', default=0.01, type=float,
                             help='正则化的权重衰减系数')
    
    return parser

def main(args):
    logs_path = args.output_log
    cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
    dataset_name = (args.dataset_path).split('/')[-1]
    # 存放训练，测试，评价日志的地址
    process_logs_path = os.path.join(logs_path, cur_time + '_Dataset->' + dataset_name)

    args.phase = 'train'
    if not os.path.exists(process_logs_path):
        os.makedirs(process_logs_path)
    else:
        print("创建存放日志文件夹错误！")
    
    # 训练，测试，评价日志
    log_train = get_logger(process_logs_path, 'train')
    log_test = get_logger(process_logs_path, 'test')
    log_eval = get_logger(process_logs_path, 'eval')

    # 将训练配置参数记录到日志里
    log_train.info("args -> " + str(args))
    log_train.info("args: dataset -> " + str(args.dataset_path))
    log_train.info("args: bce_weight -> " + str(args.bce_weight))
    log_train.info("args: iou_weight -> " + str(args.iou_weight))
    log_train.info('使用配置为：')
    log_train.info(f'输入图像尺寸 -> {args.load_width}x{args.load_height}')
    log_train.info(f'优化器 -> {args.optimizer}')
    log_train.info(f'训练总批次 -> {args.epochs}')
    log_train.info(f'学习率更新策略 -> {args.lr_scheduler}')
    log_train.info(f'初始学习率 -> {args.lr}')
    log_train.info(f'权重衰减 -> {args.weight_decay}')
    # 打印在终端的信息
    print('使用配置为：')
    print(f'bce_weight -> {args.bce_weight}')
    print(f'iou_weight -> {args.iou_weight}')
    print(f'dataset -> {args.dataset_path}')
    print(f'输入图像尺寸 -> {args.load_width}x{args.load_height}')
    print(f'优化器 -> {args.optimizer}')
    print(f'训练总批次 -> {args.epochs}')
    print(f'学习率更新策略 -> {args.lr_scheduler}')
    print(f'初始学习率 -> {args.lr}')
    print(f'权重衰减 -> {args.weight_decay}')


    # 设置训练设备和随机种子
    device = torch.device(args.device)
    # 后面的是为了分布式训练时每个进程的种子有差异，单进程时后面函数返回0
    seed = args.seed + utils.get_rank() 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 创建模型
    model, criterion = build_model(args)
    model.to(device)
    args.batch_size = args.batch_size_train
    train_dataLoader = create_dataset(args)
    dataset_size = len(train_dataLoader)
    print('训练数据集大小 = %d' % dataset_size)
    log_train.info('训练数据集大小 = %d' % dataset_size)

    # 用来给AdamW优化器设置参数，可传入只需要梯度更新的参数
    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters() if p.requires_grad],
            "lr": args.lr,
        },
    ]

    # 设置优化器
    if args.optimizer == 'sgd':
        print('使用SGD优化器!')
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        print('使用AdamW优化器!')
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    
    # 设置学习率调度器
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'CosLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    elif args.lr_scheduler == 'PolyLR':
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
    max_mIoU = 0
    # 用于记录综合评价指标（可根据 --eval_metric1/--eval_metric2 选择）
    max_eval = -1.0
    max_Metrics = {'epoch': 0, 'mIoU': 0, 'ODS': 0, 'OIS': 0, 'F1': 0, 'Precision': 0, 'Recall': 0}

    for epoch in range(args.start_epoch, args.epochs):
        print("---------------------------------------------------------------------------------------")
        print("开始训练 epoch -> ", epoch)
        # 训练一个 epoch
        train_one_epoch(model, criterion, train_dataLoader, optimizer, epoch, args, log_train)
        # 更新学习率
        lr_scheduler.step()
        print("结束训练 epoch -> ", epoch)
        print("---------------------------------------------------------------------------------------")

        print("开始测试 epoch -> ", epoch)
        # 保存测试数据的地址
        results_path = cur_time + '_Dataset->' + dataset_name
        save_root = f'{args.output_results}/{results_path}/results_' + str(epoch)
        # 修改成测试阶段参数
        args.phase = 'test'
        args.batch_size = args.batch_size_test
        test_dl = create_dataset(args)
        pbar = tqdm(total=len(test_dl), desc=f"Initial Loss: Pending")

        if not os.path.isdir(save_root):
            os.makedirs(save_root)
            
        with torch.no_grad():
            model.eval()
            for batch_idx, (data) in enumerate(test_dl):
                x = data["image"].to(device)
                target = data["label"].to(device)
                # 只需要Mask1
                out, _, _ = model(x)
                loss = criterion(out, target.float())

                # 逐样本保存 batch 内所有图像（适配任意 batch_size）
                B = out.shape[0]
                for b in range(B):
                    t = target[b, 0, ...].cpu().numpy()
                    o = out[b, 0, ...].cpu().numpy()
                    # 从路径里取对应文件名
                    root_name = data["A_paths"][b].split("/")[-1][0:-4]
                    # 安全归一化，避免除以 0
                    t_max = np.max(t) if np.max(t) > 0 else 1.0
                    o_max = np.max(o) if np.max(o) > 0 else 1.0
                    t_img = (255.0 * (t / t_max)).astype(np.uint8)
                    o_img = (255.0 * (o / o_max)).astype(np.uint8)
                    
                    log_test.info('----------------------------------------------------------------------------------------------')
                    log_test.info("loss -> " + str(loss))
                    log_test.info(str(os.path.join(save_root, "{}_lab.png".format(root_name))))
                    log_test.info(str(os.path.join(save_root, "{}_pre.png".format(root_name))))
                    log_test.info('----------------------------------------------------------------------------------------------')
                    cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), t_img)
                    cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), o_img)

                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)
        pbar.close()

        log_test.info("model -> " + str(epoch) + " test finish!")
        log_test.info('----------------------------------------------------------------------------------------------')
        print("结束测试 epoch -> ", epoch)
        print("---------------------------------------------------------------------------------------")

        print("开始评估 epoch -> ", epoch)
        metrics = eval(log_eval, save_root, epoch)
        for key, value in metrics.items():
            print(str(key) + ' -> ' + str(value))
        # 使用通过命令行指定的两个评价指标进行比较（例如 ODS + OIS）
        eval_val = metrics.get(args.eval_metric1, 0) + metrics.get(args.eval_metric2, 0)
        if max_eval < eval_val:
            max_Metrics = metrics
            max_eval = eval_val
            # 保存最佳模型参数
            checkpoint_paths = [output_dir / f'checkpoint_best.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            log_train.info("\n更新并保存最佳模型 -> " + str(epoch))
            print("\n更新并保存最佳模型 -> ", epoch)

        print("结束评估 epoch -> ", epoch)
        # 根据选择的评价指标格式化输出：若为 mIoU 单指标，则显示 max_mIoU；
        # 若为其他组合（例如 ODS+OIS），则显示组合名及其相加的值
        if args.eval_metric1 == 'mIoU' and args.eval_metric2 == 'mIoU':
            print('\nmax_mIoU -> ' + str(max_Metrics['mIoU']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        else:
            combo_name = f"{args.eval_metric1}+{args.eval_metric2}"
            combo_value = max_Metrics.get(args.eval_metric1, 0) + max_Metrics.get(args.eval_metric2, 0)
            print(f"\nmax_{combo_name} -> {combo_value}\nmax Epoch -> " + str(max_Metrics['epoch']))
        print("---------------------------------------------------------------------------------------")

        log_eval.info("evalauting epoch finish -> " + str(epoch))
        if args.eval_metric1 == 'mIoU' and args.eval_metric2 == 'mIoU':
            log_eval.info('\nmax_mIoU -> ' + str(max_Metrics['mIoU']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        else:
            combo_name = f"{args.eval_metric1}+{args.eval_metric2}"
            combo_value = max_Metrics.get(args.eval_metric1, 0) + max_Metrics.get(args.eval_metric2, 0)
            log_eval.info(f"\nmax_{combo_name} -> {combo_value}\nmax Epoch -> " + str(max_Metrics['epoch']))
        log_eval.info("---------------------------------------------------------------------------------------")

    # 把最佳评价得分写入日志
    for key, value in max_Metrics.items():
        log_eval.info(str(key) + ' -> ' + str(value))
        print(str(key) + ' -> ' + str(value))
    if args.eval_metric1 == 'mIoU' and args.eval_metric2 == 'mIoU':
        log_eval.info('\nmax_mIoU -> ' + str(max_Metrics['mIoU']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
    else:
        combo_name = f"{args.eval_metric1}+{args.eval_metric2}"
        combo_value = max_Metrics.get(args.eval_metric1, 0) + max_Metrics.get(args.eval_metric2, 0)
        log_eval.info(f"\nmax_{combo_name} -> {combo_value}\nmax Epoch -> " + str(max_Metrics['epoch']))
        print(f"\nmax_{combo_name} -> {combo_value}\nmax Epoch -> " + str(max_Metrics['epoch']))

    print("---------------------------------------------------------------------------------------")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('使用配置为：')
    print(f'bce_weight -> {args.bce_weight}')
    print(f'iou_weight -> {args.iou_weight}')
    print(f'dataset -> {args.dataset_path}')
    print(f'输入图像尺寸 -> {args.load_width}x{args.load_height}')
    print(f'优化器 -> {args.optimizer}')
    print(f'训练总批次 -> {args.epochs}')
    print(f'学习率更新策略 -> {args.lr_scheduler}')
    print(f'初始学习率 -> {args.lr}')
    print(f'权重衰减 -> {args.weight_decay}')
    print(f'模型更新评价指标1 -> {args.eval_metric1}')
    print(f'模型更新评价指标2 -> {args.eval_metric2}')
    log_train.info("args -> " + str(args))
    log_train.info("args: dataset -> " + str(args.dataset_path))
    log_train.info("args: bce_weight -> " + str(args.bce_weight))
    log_train.info("args: iou_weight -> " + str(args.iou_weight))
    log_train.info('使用配置为：')
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
