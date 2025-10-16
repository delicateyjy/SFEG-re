import numpy as np
import os
import logging
import glob
import cv2
import torch
from tqdm import tqdm
from pathlib import Path

def get_statistics(pred, gt):
    """计算单张图像的TP, FP, FN"""
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return tp, fp, fn

def cal_mIoU_metrics(pred_list, gt_list, thresh_step=0.01):
    """寻找最优阈值下的mIoU"""
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            
            tp = np.sum((pred_img == 1) & (gt_img == 1))
            fp = np.sum((pred_img == 1) & (gt_img == 0))
            fn = np.sum((pred_img == 0) & (gt_img == 1))
            tn = np.sum((pred_img == 0) & (gt_img == 0))

            iou_foreground = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0
            iou_background = tn / (tn + fp + fn) if (tn + fp + fn) != 0 else 0
            
            iou_list.append((iou_foreground + iou_background) / 2)

        final_iou.append(np.mean(iou_list))
        
    return np.max(final_iou)

def cal_ODS_metrics(pred_list, gt_list, thresh_step=0.01):
    """计算ODS指标: 全局最优阈值下的F1, P, R"""
    metrics_by_thresh = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        # 累积整个数据集的TP, FP, FN
        total_tp, total_fp, total_fn = 0, 0, 0
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        # 基于全局统计值计算P, R, F1
        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        metrics_by_thresh.append({'f1': f1, 'p': p, 'r': r, 'thresh': thresh})

    # 找到F1分数最高的指标
    best_metrics = max(metrics_by_thresh, key=lambda x: x['f1'])
    return best_metrics['f1'], best_metrics['p'], best_metrics['r'], best_metrics['thresh']

def cal_OIS_metrics(pred_list, gt_list, thresh_step=0.01):
    """计算OIS指标: 单图最优阈值的平均F1, P, R"""
    best_f1_list, best_p_list, best_r_list = [], [], []

    for pred, gt in zip(pred_list, gt_list):
        metrics_by_thresh = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            metrics_by_thresh.append({'f1': f1, 'p': p, 'r': r})
        
        # 找到单张图像的最优指标
        best_image_metric = max(metrics_by_thresh, key=lambda x: x['f1'])
        best_f1_list.append(best_image_metric['f1'])
        best_p_list.append(best_image_metric['p'])
        best_r_list.append(best_image_metric['r'])

    # 对所有图像的最优结果求平均
    return np.mean(best_f1_list), np.mean(best_p_list), np.mean(best_r_list)

def imread(path, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
    """读取图像的辅助函数"""
    im = cv2.imread(path, load_mode)
    if im is None:
        raise FileNotFoundError(f"未找到图像: {path}")
    if convert_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if load_size > 0:
        im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    if thresh > 0:
        _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    return im

def get_image_pairs(data_dir, suffix_gt='real_B', suffix_pred='fake_B'):
    """根据后缀获取预测图和真值图标注"""
    gt_list = glob.glob(os.path.join(data_dir, f'*{suffix_gt}.png'))
    if not gt_list:
        print(f"警告: 在'{data_dir}'中未找到后缀为'*{suffix_gt}.png'的真值图")
        return [], [], [], []
    pred_list = [ll.replace(suffix_gt, suffix_pred) for ll in gt_list]
    
    pred_imgs, gt_imgs, pred_imgs_names, gt_imgs_names = [], [], [], []
    for pred_path, gt_path in zip(pred_list, gt_list):
        if not os.path.exists(pred_path):
            print(f"警告: 未找到 {gt_path} 对应的预测图, 已跳过。")
            continue
        pred_imgs.append(imread(pred_path))
        gt_imgs.append(imread(gt_path, thresh=127))
        pred_imgs_names.append(pred_path)
        gt_imgs_names.append(gt_path)
    return pred_imgs, gt_imgs, pred_imgs_names, gt_imgs_names

def eval(log_eval, results_dir, epoch):
    """主评估函数。返回：epoch，mIoU，ODS_F1，ODS_P，ODS_R，OIS_F1，OIS_P，OIS_R"""
    suffix_gt = "lab"
    suffix_pred = "pre"
    log_eval.info(f"评估目录: {results_dir}, 轮次: {epoch}")

    src_img_list, tgt_img_list, _, _ = get_image_pairs(results_dir, suffix_gt, suffix_pred)
    
    if not src_img_list:
        log_eval.warning("未找到评估图像对，跳过本次评估。")
        return {'epoch': epoch}

    mIoU = cal_mIoU_metrics(src_img_list, tgt_img_list)
    ods_f1, ods_p, ods_r = cal_ODS_metrics(src_img_list, tgt_img_list)
    ois_f1, ois_p, ois_r = cal_OIS_metrics(src_img_list, tgt_img_list)

    log_eval.info(f"mIoU -> {mIoU:.4f}")
    log_eval.info(f"ODS_F1 -> {ods_f1:.4f}, ODS_P -> {ods_p:.4f}, ODS_R -> {ods_r:.4f}")
    log_eval.info(f"OIS_F1 -> {ois_f1:.4f}, OIS_P -> {ois_p:.4f}, OIS_R -> {ois_r:.4f}")
    log_eval.info("评估完成!")

    return {
        'epoch': epoch,
        'mIoU': mIoU,
        'ODS_F1': ods_f1,
        'ODS_P': ods_p,
        'ODS_R': ods_r,
        'OIS_F1': ois_f1,
        'OIS_P': ois_p,
        'OIS_R': ois_r,
    }

def evaluate_online(model, data_loader, device, epoch, save_path=None, model_outputs_logits=True):
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
    model.eval()
    all_pred_probs = []
    all_gt_maps = []
    all_filenames = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"Validating Epoch {epoch + 1}"):
            x = data["image"].to(device)
            target = data["label"].to(device)
            raw_output = model(x)
            if model_outputs_logits:
                pred_probs = torch.sigmoid(raw_output)
            else:
                pred_probs = raw_output
            target[target > 0] = 1
            all_pred_probs.extend([p.cpu() for p in pred_probs])
            all_gt_maps.extend([g.cpu() for g in target])
            if "A_paths" in data:
                all_filenames.extend([Path(p).stem for p in data["A_paths"]])
    if not all_pred_probs:
        print("警告：评估数据为空，跳过指标计算和图片保存。")
        return {'epoch': epoch + 1}
    pred_numpy_list = [(p[0].numpy() * 255) for p in all_pred_probs]
    gt_numpy_list = [(g[0].numpy() * 255) for g in all_gt_maps]
    mIoU = cal_mIoU_metrics(pred_numpy_list, gt_numpy_list)
    ods_f1, ods_p, ods_r, ods_threshold = cal_ODS_metrics(pred_numpy_list, gt_numpy_list)
    ois_f1, ois_p, ois_r = cal_OIS_metrics(pred_numpy_list, gt_numpy_list)
    if save_path:
        print(f"\n正在使用全局最佳阈值 {ods_threshold:.4f} 保存预测图至: {save_path}")
        for i in tqdm(range(len(pred_numpy_list)), desc="Saving images"):
            root_name = all_filenames[i] if i < len(all_filenames) else f"image_{i:04d}"
            prob_map_uint8 = (pred_numpy_list[i]).astype(np.uint8)
            binary_map_uint8 = (prob_map_uint8 > (ods_threshold * 255)).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(save_path, f"{root_name}_pre.png"), binary_map_uint8)
            gt_map_uint8 = (gt_numpy_list[i]).astype(np.uint8)
            cv2.imwrite(os.path.join(save_path, f"{root_name}_lab.png"), gt_map_uint8)
            cv2.imwrite(os.path.join(save_path, f"{root_name}_prob.png"), prob_map_uint8)
    return {
        'epoch': epoch + 1, 'mIoU': mIoU,
        'ODS_F1': ods_f1, 'ODS_P': ods_p, 'ODS_R': ods_r,
        'OIS_F1': ois_f1, 'OIS_P': ois_p, 'OIS_R': ois_r,
        'ODS_Threshold': ods_threshold
    }

if __name__ == '__main__':
    # 配置日志记录器以便在控制台显示信息
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # !! 请确保此处的路径正确 !!
    results_dir = "results/2025_10_09_14:02:18_Dataset->CrackMap/results_9"
    epoch_identifier = "standalone_test"

    # 调用统一的评估函数
    evaluation_results = eval(logging, results_dir, epoch_identifier)

    # 打印评估结果
    print("\n--- 独立评估结果摘要 ---")
    if evaluation_results and len(evaluation_results) > 1:
        for key, value in evaluation_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    else:
        print("未能完成评估。")

