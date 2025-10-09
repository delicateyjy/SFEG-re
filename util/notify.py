import os
import requests
from typing import Optional, Dict

# AutoDL的 Token
AUTODL_WECHAT_TOKEN = "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjc3ODczNywidXVpZCI6ImIwMmE4ZjBjMDY4NGU5ZDYiLCJpc19hZG1pbiI6ZmFsc2UsImJhY2tzdGFnZV9yb2xlIjoiIiwiaXNfc3VwZXJfYWRtaW4iOmZhbHNlLCJzdWJfbmFtZSI6IiIsInRlbmFudCI6ImF1dG9kbCIsInVwayI6IiJ9.pu4LEDW3kx8qhsDxSL6KlDP1j7Vfrz8rsvYg0hYDpS687tFxM-lGevWgknGqOjGBcb9S2SBncF8_m__iOUS2tg"

def send_training_completion_notification(
    final_metrics: Dict,
    dataset_name: str,
    early_stopper: Optional[object] = None,
    total_epochs: Optional[int] = None,
    eval_metric1: Optional[str] = 'mIoU',
    eval_metric2: Optional[str] = 'ODS',
) -> None:
    if not AUTODL_WECHAT_TOKEN:
        print("未配置Token，跳过微信通知。")
        return

    if early_stopper and early_stopper.stopped_epoch > 0:
        summary = early_stopper.get_summary()
        title = f"早停 {dataset_name}"
        name = f"实验 - {summary.get('monitor_metric', '复合指标')}"
        content = (
            f"最佳Epoch: {summary.get('best_epoch', 'N/A')}\n"
            f"最佳分数: {summary.get('best_score', 'N/A'):.6f}\n"
            f"停止于Epoch: {summary.get('stopped_epoch', 'N/A')}\n\n"
            f"--- 详细指标 ---\n"
            f"mIoU: {final_metrics.get('mIoU', 'N/A'):.4f}\n"
            f"ODS: {final_metrics.get('ODS', 'N/A'):.4f}\n"
            f"OIS: {final_metrics.get('OIS', 'N/A'):.4f}\n"
            f"F1: {final_metrics.get('F1', 'N/A'):.4f}\n"
            f"Precision: {final_metrics.get('AP', 'N/A'):.4f}\n"
            f"Recall: {final_metrics.get('AR', 'N/A'):.4f}\n"
        )
    else:
        combo_name = f"{eval_metric1}+{eval_metric2}"
        combo_value = final_metrics.get(eval_metric1, 0) + final_metrics.get(eval_metric2, 0)
        title = f"完成 {dataset_name}"
        name = f"实验 - {combo_name}"
        content = (
            f"最佳Epoch: {final_metrics.get('epoch', 'N/A')}\n"
            f"在 {total_epochs} 轮训练后达到最佳\n\n"
            f"--- 详细指标 ---\n"
            f"{combo_name}: {combo_value:.6f}\n"
            f"mIoU: {final_metrics.get('mIoU', 'N/A'):.4f}\n"
            f"ODS: {final_metrics.get('ODS', 'N/A'):.4f}\n"
            f"OIS: {final_metrics.get('OIS', 'N/A'):.4f}\n"
            f"F1: {final_metrics.get('F1', 'N/A'):.4f}\n"
            f"Precision: {final_metrics.get('AP', 'N/A'):.4f}\n"
            f"Recall: {final_metrics.get('AR', 'N/A'):.4f}\n"
        )
        
    url = "https://www.autodl.com/api/v1/wechat/message/send"
    headers = {"Authorization": AUTODL_WECHAT_TOKEN}
    try:
        requests.post(url, json={"title": title, "name": name, "content": content}, headers=headers, timeout=5)
        print("微信通知已尝试发送。")
    except Exception as e:
        print(f"发送微信通知时发生网络错误: {e}")
