import time
import swanlab
import torch
import numpy as np
from tqdm import tqdm
from thop import profile
import argparse
import os
import logging
import datetime
import random
import ml_collections
import copy

from dataset import create_dataset
from util.logger import get_logger
from util.early_stopping import EarlyStopping
from eval.evaluate import evaluate_online
import util.misc as utils

# 导入模型和损失函数，根据选择的模型动态导入

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 设置参数
parser = argparse.ArgumentParser('crackdection')
parser.add_argument('--train_model', type=str, 
                    choices=['Crackformer2', 'CDSNet', 'CTCrackSeg', 'UCTransNet', 'Crackmer'], default='Crackmer')
parser.add_argument('--dataset_path', default="/home/lab/Code/data/CRACK500")
parser.add_argument('--dataset_mode', type=str, default='crack')
parser.add_argument('--load_width', type=int, default=512)
parser.add_argument('--load_height', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_threads', type=int, default=0)
parser.add_argument('--serial_batches', type=bool, default=False)
parser.add_argument('--patience', type=int, default=20)

args = parser.parse_args()

# 创建不同模型以及对应的配置
if args.train_model.lower() == 'crackformer2':
    # Crackformer2
    from CrackDection.Crackformer2 import crackformer2, crackformer2_loss
    model, criterion = crackformer2().to(device), crackformer2_loss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 100], gamma=0.1)
    args.epochs = 500
elif args.train_model.lower() == 'cdsnet':
    # CDSNet
    from CrackDection.CDSNet import CDSNET, exfloss
    model, criterion = CDSNET(in_channels=3, out_channels=1).to(device), exfloss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
    args.epochs = 100
elif args.train_model.lower() == 'ctcrackseg':
    # CTCrackSeg
    from CrackDection.CTCrackSeg import CTCrackSeg, DiceBCELoss
    model, criterion = CTCrackSeg().to(device), DiceBCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=6)
    args.epochs = 100
    args.batch_size = 2 # 原作者使用的2，跑不通，尝试改为1，仍然跑不通，只能用Autodl了
elif args.train_model.lower() == 'uctransnet':
    # UCTransNet
    from CrackDection.UCtransNet import UCTransNet, WeightedDiceBCE
    def get_CTranS_config():
        config = ml_collections.ConfigDict()
        config.transformer = ml_collections.ConfigDict()
        config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
        config.transformer.num_heads = 4
        config.transformer.num_layers = 4
        config.expand_ratio = 4  # MLP channel dimension expand ratio
        config.transformer.embeddings_dropout_rate = 0.1
        config.transformer.attention_dropout_rate = 0.1
        config.transformer.dropout_rate = 0
        config.patch_sizes = [16, 8, 4, 2]
        config.base_channel = 64  # base channel of U-Net
        config.n_classes = 1
        return config
    config_vit = get_CTranS_config()
    model = UCTransNet(config_vit,n_channels=3,n_classes=1).to(device)
    criterion = WeightedDiceBCE(dice_weight=0.5,BCE_weight=0.5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    args.epochs = 300
    args.patience = 50
elif args.train_model.lower() == 'crackmer':
    # Crackmer
    from CrackDection.Crackmer import Net, Loss
    model, criterion = Net().to(device), Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-4)
    args.epochs = 80
    args.batch_size = 2 # 原作者使用4，显存不够，改为2
else:
    raise ValueError(f"未知的模型类型: {args.train_model}")

model.to(device)

# 存放模型参数，日志，结果的路径
cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
output_dir = f'output/{args.train_model}/{os.path.basename(args.dataset_path)}/{cur_time}/'
os.makedirs(output_dir, exist_ok=True)
log = get_logger(output_dir, 'train')

# 计算模型计算量等
dummy_input = torch.randn(1, 3, args.load_height, args.load_width, device=device)
flops, params = profile(copy.deepcopy(model), inputs=(dummy_input,), verbose=False)
args.gflops = f"{flops / 1e9:.2f}G"
args.m_params = f"{params / 1e6:.2f}M"
print(f"Model Complexity: {args.gflops} FLOPs, {args.m_params} Params")
log.info(f"Model Complexity: {args.gflops} FLOPs, {args.m_params} Params")

# 创建训练集
args.phase = 'train'
train_dataLoader = create_dataset(args)
print(f'训练数据集大小 = {len(train_dataLoader)}')
log.info(f'训练数据集大小 = {len(train_dataLoader)}')

# 创建验证集
args.phase = 'val'
args.batch_size = 1  # 验证时batch size均设为1
val_dataLoader = create_dataset(args)
print(f'验证数据集大小 = {len(val_dataLoader)}')
log.info(f'验证数据集大小 = {len(val_dataLoader)}')

# 创建测试集
args.phase = 'test'
test_dataLoader = create_dataset(args)
print(f'测试数据集大小 = {len(test_dataLoader)}')
log.info(f'测试数据集大小 = {len(test_dataLoader)}')

# 早停
monitor_name = 'ODS_F1+OIS_F1'
early_stopper = EarlyStopping(
    patience=args.patience,
    min_delta=0.0005,
    monitor=monitor_name,
    lr_patience=10,
    lr_factor=0.5,
    restore_best_weights=True,
    mode='max'  # ODS, OIS, mIoU都是越大越好
)
print(f"早停机制已启用: 监控 '{monitor_name}', 耐心值={early_stopper.patience}")
log.info(f"早停机制已启用: 监控 '{monitor_name}', 耐心值={early_stopper.patience}")

start_time = time.time()
max_Metrics = {'epoch': -1, 'mIoU': 0, 'ODS_F1': 0, 'ODS_P': 0, 'ODS_R': 0, 'OIS_F1': 0, 'OIS_P': 0, 'OIS_R': 0}

# 添加种子
args.seed = 42
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


swanlab.init(
    project="CrackDection",
    name=f"{args.train_model}_{os.path.basename(args.dataset_path)}",
    description="",
    workspace="crack_segmentation",
    config={
        'model': args.train_model,
        'dataset': os.path.basename(args.dataset_path),
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'optimizer': optimizer.__class__.__name__,
        'scheduler': scheduler.__class__.__name__,
        'init_lr': optimizer.param_groups[0]['lr'],
        'input_size': (args.load_height, args.load_width),
        'GFLOPs': args.gflops,
        'M_Params': args.m_params,
    }
)

print('---------------------- 开始训练 ----------------------')
log.info('---------------------- 开始训练 ----------------------')
for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    train_bar = tqdm(train_dataLoader, desc=f'Epoch {epoch + 1}/{args.epochs} Training')
    for i, data in enumerate(train_dataLoader):
        samples = data['image'].to(device)
        targets = data['label'].to(device)

        # CTCrackSeg的数据集含有boundary，我们的没有，就直接用输出的Mask计算损失
        Mask = model(samples)
        Mask = Mask.float()

        targets[targets > 0] = 1
        targets = targets.float()

        loss = criterion(Mask, targets)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_bar.set_description(f"Loss: {loss.item():.4f}")
        train_bar.update(1)
    
    avg_train_loss = train_loss / len(train_dataLoader)
    print(f'Epoch {epoch + 1}/{args.epochs}, Training Loss: {avg_train_loss:.4f}')
    log.info(f'Epoch {epoch + 1}/{args.epochs}, Training Loss:{avg_train_loss:.4f}')
    swanlab.log({'train_loss': avg_train_loss, 'epoch': epoch})
    train_bar.close()

    # 早停判断是否更新学习率
    # 训练CTCrackSeg时，调度器放在验证阶段
    if not (early_stopper and early_stopper.lr_patience < early_stopper.patience):
        scheduler.step()

    # 每个epoch后进行验证
    print(f'---------------------- 第 {epoch + 1} 轮验证 ----------------------')
    log.info(f'---------------------- 第 {epoch + 1} 轮验证 ----------------------')
    model.eval()
    val_metrics = evaluate_online(model, val_dataLoader, device, epoch)
    for key, value in val_metrics.items():
        if isinstance(value, float):
            print(f'Validation {key} -> {value:.4f}')
            log.info(f"Epoch {epoch} | Validation {key} -> {value:.4f}")
        swanlab.log({f'val_{key}': value, 'epoch': epoch})
    swanlab.log({'current_lr': optimizer.param_groups[0]['lr'], 'epoch': epoch})
    
    # 随机抽一张图像进行可视化并上传swanlab
    vis_data = next(iter(val_dataLoader))
    vis_image, vis_label = vis_data['image'].to(device), vis_data['label'].to(device)
    with torch.no_grad():
        vis_pred = (model(vis_image)[0] > 0.5).float()

    img_np = vis_image[0].cpu().permute(1, 2, 0).numpy()
    combined_image = np.hstack([((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255), vis_label[0].cpu().permute(1, 2, 0).repeat(1, 1, 3).numpy() * 255, vis_pred.cpu().permute(1, 2, 0).repeat(1, 1, 3).numpy() * 255]).astype(np.uint8)
    swanlab.log({"predictions": swanlab.Image(combined_image, caption=f"Epoch-{epoch}"), "epoch": epoch})

    # 使用早停保存模型参数
    eval_val = val_metrics.get('ODS_F1', 0) + val_metrics.get('OIS_F1', 0)

    # 训练CTCrackSeg时，调度器放在验证阶段，其他模型将以下代码注释
    # scheduler.step(eval_val)

    if early_stopper:
        should_stop = early_stopper(
            current_score=eval_val, epoch=epoch, model=model, optimizer=optimizer,
            lr_scheduler=scheduler, output_dir=output_dir, logger=log,
            extra_save_info={'args': args}
        )
        if early_stopper.best_epoch == epoch:
            max_Metrics = val_metrics
        if should_stop:
            print("早停条件已满足，训练提前结束。")
            log.info("早停条件已满足，训练提前结束。")
            break
    else: # 无早停时的逻辑
        current_max_val = max_Metrics.get('ODS_F1', 0) + max_Metrics.get('OIS_F1', 0)
        if eval_val > current_max_val:
            max_Metrics = val_metrics
            checkpoint_path = output_dir / 'checkpoint_best.pth'
            utils.save_on_master({
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(), 'epoch': epoch, 'args': args,
            }, checkpoint_path)
            log.info(f"\n更新并保存最佳模型 -> {epoch}")
            print(f"\n更新并保存最佳模型 -> {epoch}")

# 把训练时的最佳验证得分记录
print("\n--- 训练过程中的最佳验证性能 ---")
log.info("--- 训练过程中的最佳验证性能 ---")
best_metrics_log = {}
for key, value in max_Metrics.items():
    log_key = f'best_val_{key}'
    if isinstance(value, float):
        print(f'Best Validation {key} -> {value:.4f}')
        log.info(f'Best Validation {key} -> {value:.4f}')
        best_metrics_log[log_key] = value
    else:
        print(f'Best Validation Epoch -> {value}')
        log.info(f'Best Validation Epoch -> {value}')
if best_metrics_log:
    swanlab.log(best_metrics_log)

# 训练完成后进行测试
print('---------------------- 测试 ----------------------')
log.info('---------------------- 测试 ----------------------')
best_model_path = os.path.join(output_dir, 'checkpoint_best.pth')
checkpoint = torch.load(best_model_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
best_epoch = checkpoint['epoch']
print(f"已加载 Epoch {best_epoch} 的最佳模型进行最终测试。")
log.info(f"已加载 Epoch {best_epoch} 的最佳模型进行最终测试。")
final_results_path = os.path.join(output_dir, 'final_test_results')
final_test_metrics = evaluate_online(
    model=model,
    data_loader=test_dataLoader,
    device=device,
    epoch=best_epoch,
    save_path=final_results_path
)
test_metrics_log = {}
print("\n--- 最终模型在测试集上的性能 ---")
log.info("--- 最终模型在测试集上的性能 ---")
for key, value in final_test_metrics.items():
    log_key = f'test_{key}'
    if isinstance(value, float):
        print(f'Test {key} -> {value:.4f}')
        log.info(f'Test {key} -> {value:.4f}')
        test_metrics_log[log_key] = value
if test_metrics_log:
    swanlab.log(test_metrics_log)

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
    log.info(summary_str)

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print(f'\n代码运行时间为 {total_time_str}')
log.info(f'代码运行时间为 {total_time_str}')
