配置环境
安装cuda11.7版本（似乎不支持cuFFT函数:RuntimeError: cuFFT error: CUFFT_INTERNAL_ERROR）
```shell
conda create -n mmcls python=3.10 -y
conda activate mmcls
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim 
mim install mmcv-full==1.7.2  
# pip install mamba-ssm==1.2.0 # 不使用mamba时注释
pip install timm lmdb mmengine thop numpy==1.26.4 opencv-python==4.8.1.78
```

创建新的环境
```shell
conda create -n mmcls_stable python=3.10 -y
conda activate mmcls_stable
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmcv-full==1.7.2  
pip install timm lmdb mmengine thop numpy==1.26.4 opencv-python==4.8.1.78

# # 可选，使用swanlab追踪训练 https://docs.swanlab.cn/
# # 与swanlab冲突，删除后不影响mmcv
# pip uninstall --yes openmim openxlab opendatalab rich 
# pip install swanlab
```
