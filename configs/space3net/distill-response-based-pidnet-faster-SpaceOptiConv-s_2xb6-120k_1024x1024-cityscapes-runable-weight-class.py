_base_ = [
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py'
]

# ==================================================================
# 1. 保留你的所有变量定义
# ==================================================================
class_weight = [
    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786,
    1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
    1.0507
]

# 你的 Student 预训练权重
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'
crop_size = (1024, 1024)

# 你的数据预处理 (移动到最外层模型，供 Teacher 和 Student 共用)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

norm_cfg = dict(type='SyncBN', requires_grad=True)

# ==================================================================
# 2. 定义 Teacher (PIDNet-L) - 新增部分
# ==================================================================
teacher_cfg = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='PIDNet',  # 标准 PIDNet
        in_channels=3,
        channels=64,    # L 版本通道
        ppm_channels=112,
        num_stem_blocks=3,
        num_branch_blocks=4,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True)),
    decode_head=dict(
        type='PIDHead',
        in_channels=256,
        channels=256,
        num_classes=19,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        # Teacher 的 Loss 仅用于占位，不参与计算
        loss_decode=[
            dict(type='OhemCrossEntropy', loss_weight=0.4),
            dict(type='OhemCrossEntropy', loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ==================================================================
# 3. 定义 Student (你的原模型) - 仅改名为 student_cfg
# ==================================================================
# 注意：删除了 data_preprocessor，因为移到了最外层
student_cfg = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ESPIDNet',
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
        # === 你的原始 Loss 配置 (完全未动) ===
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                # class_weight=class_weight,
                loss_weight=0.4),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ==================================================================
# 4. 组装蒸馏模型 (Distiller) - 替换原本的 model
# ==================================================================
model = dict(
    type='PIDNetDistiller',
    data_preprocessor=data_preprocessor, # 使用你定义的预处理
    teacher_cfg=teacher_cfg,
    student_cfg=student_cfg,
    
    # Teacher 权重 (PIDNet-L Cityscapes)
    teacher_pretrained='work_dirs/pidnet-best/pidnet-l_2xb6-120k_1024x1024-cityscapes_20230303_114514-0783ca6b.pth',
    
    distill_weight=5.0, # 蒸馏权重
    temperature=4.0
)

# ==================================================================
# 5. 其他配置 (完全保持你原样)
# ==================================================================
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

# ================= 修改开始 =================

# 1. 优化器封装：开启梯度裁剪 (关键！把 clip_grad=None 改掉)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper', 
    optimizer=optimizer, 
    # ⚠️ 核心修改：限制梯度最大范数为 35，防止梯度爆炸导致 NaN
    clip_grad=dict(max_norm=35, norm_type=2)
)

# 2. 学习率策略：增加 Linear Warmup
param_scheduler = [
    # ⚠️ 新增：前 1000 次迭代进行预热，学习率从 0 线性增加到 0.005
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1000),
    # 原有的 Poly 策略（从 1000 iter 开始）
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=1000, 
        end=120000,
        by_epoch=False)
]

# ================= 修改结束 =================

# training schedule for 120k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=iters, val_interval=iters // 120)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=iters // 120),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

randomness = dict(seed=304)

# 你最后补充的 v1.x 修正版 pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='Pad', size_divisor=32, pad_val=0), 
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# 覆盖 dataloader 里的 pipeline
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
