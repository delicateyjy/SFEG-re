配置环境
安装cuda11.7版本
```shell
conda create -n mmcls python=3.10 -y
conda activate mmcls
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim 
mim install mmcv-full==1.7.2  
# pip install mamba-ssm==1.2.0 # 不使用mamba时注释
pip install timm lmdb mmengine thop numpy==1.26.4 opencv-python==4.8.1.78
```