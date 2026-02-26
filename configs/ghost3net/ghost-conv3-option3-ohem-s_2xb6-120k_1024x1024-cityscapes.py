# 20260225:
# ROY: 方案三：优化 OHEM（在线难例挖掘）策略	
# 核心逻辑：GhostPIDNet 的容量比原版小，如果 OHEM 过于激进（只学最难的样本），模型可能学不会。	
# 操作步骤：	
# 1. 降低阈值：将 thres 从 0.9 降到 0.7。	
# 2. 增加保留像素：将 min_kept 从 131072 增加到 200000 甚至更多。	
# 或者在前 50k iter 关闭 OHEM（使用标准 CE Loss），后期再开启。	
# 3. 调整 Class Weight：检查一下 train_log，看看哪些类 IoU 最低（通常是 wall, fence, rider）。适当手动调高这些类的 class_weight。	

_base_ = [
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py'
]

# The class_weight is borrowed from https://github.com/openseg-group/OCNet.pytorch/issues/14 # noqa
# Licensed under the MIT License
# class_weight = [
#     0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786,
#     1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
#     1.0507
# ]
# 改1
# 优化后的权重 (针对 Cityscapes 难类：Wall, Fence, Pole, Sign, Rider, Bicycle 适当增加 10%~15%)
class_weight = [
    0.8373, 0.918, 0.866, 
    1.15,   # Wall (原 1.03)
    1.15,   # Fence (原 1.01)
    1.15,   # Pole (原 0.99) - 重点
    1.10,   # Traffic Light (原 0.97)
    1.15,   # Traffic Sign (原 1.04) - 重点
    0.8786, 1.0023, 0.9539, 
    1.10,   # Rider (原 0.98) - 重点
    1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 
    1.25,   # Motorcycle (原 1.15)
    1.20    # Bicycle (原 1.05) - 重点
]

# 改1 初始化权重 
#checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'  # noqa

checkpoint_file = 'checkpoints/pidnet-improved_ghost_conv_class_weight_s_2xb6-120k_1024x1024-cityscapes/iter_240000.pth'
crop_size = (1024, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='PIDNetImprovedGhostConv',
        in_channels=3,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    decode_head=dict(
        type='PIDHead',
        in_channels=128,
        channels=128,
        num_classes=19,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                # class_weight=class_weight,
                loss_weight=0.4),
            dict(
                type='OhemCrossEntropy',
                # 改2
                thres=0.7,
                # 改3
                min_kept=220000,
                class_weight=class_weight,
                loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.7,
                min_kept=220000,
                class_weight=class_weight,
                loss_weight=1.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs')
]
train_dataloader = dict(batch_size=6, dataset=dict(pipeline=train_pipeline))

iters = 120000
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]
# training schedule for 120k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=iters, val_interval=iters // 10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=iters // 10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

randomness = dict(seed=304)
