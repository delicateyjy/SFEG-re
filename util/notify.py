import os
import requests
from typing import Optional, Dict

# AutoDL的 Token
AUTODL_WECHAT_TOKEN = "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjc3ODczNywidXVpZCI6ImIwMmE4ZjBjMDY4NGU5ZDYiLCJpc19hZG1pbiI6ZmFsc2UsImJhY2tzdGFnZV9yb2xlIjoiIiwiaXNfc3VwZXJfYWRtaW4iOmZhbHNlLCJzdWJfbmFtZSI6IiIsInRlbmFudCI6ImF1dG9kbCIsInVwayI6IiJ9.pu4LEDW3kx8qhsDxSL6KlDP1j7Vfrz8rsvYg0hYDpS687tFxM-lGevWgknGqOjGBcb9S2SBncF8_m__iOUS2tg"

def _format_metric(value, precision=4):
    """安全地格式化指标，如果值不是数字则返回'N/A'"""
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return 'N/A'

def send_training_completion_notification(
    final_metrics: Dict,
    dataset_name: str,
    early_stopper: Optional[object] = None,
    total_epochs: Optional[int] = None,
    eval_metric1: Optional[str] = 'mIoU',
    eval_metric2: Optional[str] = 'mIoU',
) -> None:
    if not AUTODL_WECHAT_TOKEN:
        print("未配置Token，跳过微信通知。")
        return

    # 先获取所有可能用到的指标值
    mIoU = final_metrics.get('mIoU')
    ODS_F1 = final_metrics.get('ODS_F1') 
    ODS_P = final_metrics.get('ODS_P')
    ODS_R = final_metrics.get('ODS_R')
    OIS_F1 = final_metrics.get('OIS_F1')
    OIS_P = final_metrics.get('OIS_P')
    OIS_R = final_metrics.get('OIS_R')
    ODS_Threshold = final_metrics.get('ODS_Threshold')

    if early_stopper and early_stopper.stopped_epoch > 0:
        summary = early_stopper.get_summary()
        title = f"早停 {dataset_name}"
        name = f"实验 - {summary.get('monitor_metric', '复合指标')}"
        content = (
            f"最佳Epoch: {summary.get('best_epoch', 'N/A')}\n"
            f"最佳分数: {_format_metric(summary.get('best_score'), 6)}\n"
            f"停止于Epoch: {summary.get('stopped_epoch', 'N/A')}\n\n"
            f"--- 详细指标 ---\n"
            f"mIoU: {_format_metric(mIoU, 4)}\n"
            f"ODS_F1: {_format_metric(ODS_F1, 4)}\n"
            f"ODS_P: {_format_metric(ODS_P, 4)}\n"
            f"ODS_R: {_format_metric(ODS_R, 4)}\n"
            f"OIS_F1: {_format_metric(OIS_F1, 4)}\n"
            f"OIS_P: {_format_metric(OIS_P, 4)}\n"
            f"OIS_R: {_format_metric(OIS_R, 4)}\n"
            f"ODS_Threshold: {_format_metric(ODS_Threshold, 4)}\n"
        )
    else:
        metric1_val = final_metrics.get(eval_metric1, 0)
        metric2_val = final_metrics.get(eval_metric2, 0)
        combo_name = f"{eval_metric1}+{eval_metric2}"
        combo_value = (metric1_val if isinstance(metric1_val, (int, float)) else 0) + \
                      (metric2_val if isinstance(metric2_val, (int, float)) else 0)
        
        title = f"完成 {dataset_name}"
        name = f"实验 - {combo_name}"
        content = (
            f"最佳Epoch: {final_metrics.get('epoch', 'N/A')}\n"
            f"在 {total_epochs} 轮训练后达到最佳\n\n"
            f"--- 详细指标 ---\n"
            f"{combo_name}: {_format_metric(combo_value, 6)}\n"
            f"mIoU: {_format_metric(mIoU, 4)}\n"
            f"ODS_F1: {_format_metric(ODS_F1, 4)}\n"
            f"ODS_P: {_format_metric(ODS_P, 4)}\n"
            f"ODS_R: {_format_metric(ODS_R, 4)}\n"
            f"OIS_F1: {_format_metric(OIS_F1, 4)}\n"
            f"OIS_P: {_format_metric(OIS_P, 4)}\n"
            f"OIS_R: {_format_metric(OIS_R, 4)}\n"
            f"ODS_Threshold: {_format_metric(ODS_Threshold, 4)}\n"
        )
        
    url = "https://www.autodl.com/api/v1/wechat/message/send"
    headers = {"Authorization": f"Bearer {AUTODL_WECHAT_TOKEN}"}
    try:
        requests.post(url, json={"title": title, "name": name, "content": content}, headers=headers, timeout=5)
        print("微信通知已尝试发送。")
    except Exception as e:
        print(f"发送微信通知时发生网络错误: {e}")

