- 0408 3090-365机 
  - 目的：测试pid head 与DDRNet-halo统一修改后的效果 python tools/train.py configs/halo_3090/halo-pidnet-s-halo-same-ddr-1xb12-120k_1024x1024-cityscapes.py --work-dir=./pidnet_workdir/halo-pidnet-s-halo-same-ddr-1xb12-120k_1024x1024-cityscapes-FULL
 
- 0407 3090-1-410机
  - 目的：1xb12 DDRNet baseline
python tools/train.py configs/halo_3090/halo-ddrnet_23-slim_in1k-pre-baseline_1xb12-120k_cityscapes-1024x1024.py --work-dir=
./ddrnet_workdir/halo-ddrnet_23-slim_in1k-pre-baseline_1xb12-120k_cityscapes-1024x1024
  - 目的：1xb12 DDRNet Smooth
configs/halo_3090/halo-ddrnet_23-slim_in1k-pre-halo_avg3-opt-smooth_1xb12-120k_cityscapes-1024x1024.py
