# 1. 显存防碎片
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 2. 启动训练
# --lr 0.00001 : 【关键修改】降低学习率，防止破坏 Epoch 1 的权重
# --batch_size 32 : 保持稳妥
python train.py \
  --name effort_universal_repro \
  --arch CLIP:ViT-L/14_svd \
  --fix_backbone \
  --use_svd \
  --data_mode wang2020 \
  --wang2020_data_path /root/autodl-tmp/datasets/CNNDetection \
  --batch_size 32 \
  --loss_freq 10 \
  --niter 20 \
  --save_epoch_freq 1 \
  --noise_std 0.02 \
  --data_aug \
  --continue_train \
  --epoch_count 0 \
  --lr 0.0002
