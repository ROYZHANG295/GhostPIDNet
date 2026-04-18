_base_ = [
    '../_base_/default_runtime.py'
]

# ================= 1. 全局变量与数据集基础设置 =================
iters = 10000  # ⚠️ 核心修复：train+val合并后图片变多，200 epoch 对应的步数变为 7800 步！
dataset_type = 'BaseSegDataset'
data_root = 'data/camvid/'
crop_size = (720, 960)

metainfo = dict(
    classes=('Sky', 'Building', 'Pole', 'Road', 'Pavement', 
             'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist'),
    palette=[[128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128], 
             [60, 40, 222], [128, 128, 0], [192, 128, 128], [64, 64, 128], 
             [64, 0, 128], [64, 64, 0], [0, 128, 192]]
)

# ⚠️ 使用 Cityscapes 预训练权重作为全局初始化
load_from = './baselines/pidnet-s_2xb6-120k_1024x1024-cityscapes_20230302_191700-bb8e3bcc.pth'

# 官方硬编码的 CamVid 类别权重 (按类别顺序)
camvid_class_weight = [
    0.58872014,  # Sky (权重极低)
    0.51052380,  # Building
    2.69662786,  # Pole (权重极高，逼迫模型去学！)
    0.45021695,  # Road (权重极低)
    1.17158675,  # Pavement
    0.77028579,  # Tree
    2.47825885,  # SignSymbol
    2.52734613,  # Fence
    1.01225269,  # Car
    3.23753095,  # Pedestrian (权重极高)
    4.13123131   # Bicyclist (全场最高权重！)
]


# ================= 2. 模型设置 =================
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
        type='PIDNet',
        in_channels=3,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True)
        # init_cfg 已经被注释掉，因为我们用全局 load_from
    ),
    decode_head=dict(
        type='PIDHead',
        in_channels=128,
        channels=128,
        num_classes=11,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=camvid_class_weight,  # ⚠️ 加上权重
                loss_weight=0.4),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                class_weight=camvid_class_weight,  # ⚠️ 加上权重
                min_kept=131072,
                loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=camvid_class_weight,  # ⚠️ 加上权重
                loss_weight=1.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ================= 3. 数据 Pipeline =================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(960, 720),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateEdge', edge_width=4), # 跑 HALO 时记得删掉这行
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(960, 720), keep_ratio=True),
    dict(type='Pad', size_divisor=32, pad_val=0), 
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# ================= 4. Dataloader 设置 (核心修改区) =================

# 定义 train 数据集
dataset_train = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    img_suffix='.png',
    seg_map_suffix='.png',
    data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
    pipeline=train_pipeline)

# 定义 val 数据集 (注意：这里使用 train_pipeline，因为它现在是作为训练集的一部分)
dataset_val_for_train = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    img_suffix='.png',
    seg_map_suffix='.png',
    data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=12,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',  # ⚠️ 核心修复：将 train 和 val 拼接起来联合训练
        datasets=[dataset_train, dataset_val_for_train]
    ))

# ⚠️ 核心修复：验证集直接指向 test 文件夹
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'), # 指向 test
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# ================= 5. 训练策略 =================
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=iters, val_interval=iters // 10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=iters // 10, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

randomness = dict(seed=304)
