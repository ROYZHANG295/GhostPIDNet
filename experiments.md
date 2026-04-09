# PIDNet-S no pretrained baseline 1xb6 seed42 3090
  - mIoU: 76.80
  - test-0325-3-2-pidnet-s-no-pretrained_baseline_warmup-2xb6-120k_1024x1024-cityscapes-seed42

# PIDNet-S no pretrained Lap Opt3 1xb6 seed42
  - mIoU: 76.69
  - test-0325-3-1-120k-pidnet-s-no-pretrained_laplacian_loss_opt3_dynamic3_warmup-2xb6-120k_1024x1024-cityscape-seed42
  - mIoU: 77.26
  - test-0317-4-1-pidnet-s-no-pretrained_warmup-2xb6-120k_1024x1024-cityscapes-seed42

# PIDNet-S with pretrained baseline 1xb12 seed304 3090
  - mIoU: 78.20
  - test-0324-1-pidnet-s-cityspace-baseline-smi-2GPUs
  - experiments/test-0324-1-120k-b12-1GPU-pidnet-s-cityspace-baseline-3090-78.20
  - experiments/test-0324-4-pidnet-s_1xb6-120k_1024x1024-cityscapes-runable-weight-class-A4000-78.09

# PIDNet-S with pretrained Lap Opt3 1xb6 seed304 3090
  - mIoU: 78.65
  - test-0322-2-120k-pidnet-s-with-pretrained_laplacian_loss_opt3_warmup-2xb6-120k_1024x1024-cityscapes
  - experiments/test-0322-2-120k-pidnet-s-with-pretrained_laplacian_loss_opt3_warmup-2xb6-120k_1024x1024-cityscapes-3090-78.65

# PIDNet-S no pretrained Lap Opt3 1xb6 seed304 3090
  - mIou: 77.06
  - test-0321-3-120k-pidnet-s-no-pretrained_laplacian_loss_opt3_warmup-2xb6-120k_1024x1024-cityscapes
  - experiments/test-0321-3-120k-pidnet-s-no-pretrained_laplacian_loss_opt3_warmup-2xb6-120k_1024x1024-cityscapes-A4000-76.41

# PIDNet-S no pretrained Lap Opt3 Dynamic1 1xb6 seed304 A4000
  - mIou=
  - experiments/test-0322-1-120k-pidnet-s-no-pretrained_laplacian_loss_opt3_dynamic_warmup-2xb6-120k_1024x1024-cityscapes-A4000-76.98

# PIDNet-S no pretrained baseline 1xb6 seed304 3090
  - mIou: 76.89 & 76.39
  - test-0320-6-pidnet-s-no-pretrained_baseline_warmup-2xb6-120k_1024x1024-cityscapes
  - experiments/test-0320-6-pidnet-s-no-pretrained_baseline_warmup-2xb6-120k_1024x1024-cityscapes-3090-76.39-76.89
  - experiments/test-0320-6-pidnet-s-no-pretrained_baseline_warmup-2xb6-120k_1024x1024-cityscapes-A4000-76.50

# PIDNet-S no pretrained Lap Dynamic3 1xb6 seed304 A4000
  - mIou=77.20
  - experiments/test-0323-2-120k-pidnet-s-no-pretrained_laplacian_loss_opt3_dynamic3_warmup-2xb6-120k_1024x1024-cityscapes-A4000-77.20

# PIDNet-S with pretrained Lap Dynamic3 1xb12 seed304 3090
  - mIoU: 78.53
  - 4-laplacian-loss-dynamic3-1GPU-120k-pidnet-s_1xb12-120k_1024x1024-cityscapes
  - experiments/4-laplacian-loss-dynamic3-1GPU-120k-pidnet-s_1xb12-120k_1024x1024-cityscapes-3090-78.53
  - experiments/4-laplacian-loss-dynamic3-1GPU-120k-pidnet-s_1xb12-120k_1024x1024-cityscapes-A4000-78.64
  
# PIDNet-S with pretrained Lap Dynamic4 1xb12 seed304 3090
  - mIoU: 78.40
  - 6-laplacian-loss-dynamic4-smooth-1GPU-120k-pidnet-s_1xb12-120k_1024x1024-cityscapes
  - experiments/6-laplacian-loss-dynamic4-smooth-1GPU-120k-pidnet-s_1xb12-120k_1024x1024-cityscapes-3090-78.40

# PIDNet-S with pretrained Lap Dynamic5 1xb12 seed304 3090
  - mIoU: 78.65
  - 7-laplacian-loss-dynamic5-smooth-1GPU-120k-pidnet-s_1xb12-120k_1024x1024-cityscapes

# BiSeNetV2 baseline 1xb6 seed304 3090
  - mIoU: 71.79
  - 10-bisenetv2_fcn_baseline-1xb6-120k_cityscapes-1024x1024

# DDRNet-23-Slim Baseline 1xb6 seed304 3090
  - mIou=76.74
  - experiments/8-ddrnet_23-slim_in1k-pre-baseline_2xb6-120k_cityscapes-1024x1024-A4000-76.74

# DDRNet-23-Slim HALO 1xb6 seed304 3090
  - mIou=77.34
  - experiments/9-1-ddrnet_23-slim_in1k-pre-halo_2xb6-120k_cityscapes-1024x1024-A4000-77.34

# 0407:
#### 目的：1xb12 DDRNet baseline
python tools/train.py configs/halo_3090/halo-ddrnet_23-slim_in1k-pre-baseline_1xb12-120k_cityscapes-1024x1024.py --work-dir=
./ddrnet_workdir/halo-ddrnet_23-slim_in1k-pre-baseline_1xb12-120k_cityscapes-1024x1024

#### 目的：1xb12 DDRNet Smooth
configs/halo_3090/halo-ddrnet_23-slim_in1k-pre-halo_avg3-opt-smooth_1xb12-120k_cityscapes-1024x1024.py