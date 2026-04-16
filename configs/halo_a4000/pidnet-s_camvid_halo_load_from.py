_base_ = [
    '../_base_/default_runtime.py'
]

# ================= 1. 数据集基础设置 (使用万能 BaseSegDataset) =================
dataset_type = 'BaseSegDataset'
data_root = 'data/camvid/'
crop_size = (720, 960)

iters = 6200

# ⚠️ 核心修复：直接在这里定义 CamVid 的类别和颜色，不需要去底层注册！
metainfo = dict(
    classes=('Sky', 'Building', 'Pole', 'Road', 'Pavement', 
             'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist'),
    palette=[[128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128], 
             [60, 40, 222], [128, 128, 0], [192, 128, 128], [64, 64, 128], 
             [64, 0, 128], [64, 64, 0], [0, 128, 192]]
)

load_from = './experiments/halo-pidnet-s-halo-same-ddr-1xb12-120k_1024x1024-cityscapes-FULL-fb_w-1_dice_w15-10-3090-79.08/best_mIoU_iter_118000.pth'

# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'  # noqa

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
        act_cfg=dict(type='ReLU', inplace=True),
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
        ),
    decode_head=dict(
        type='PIDHeadHALOSameDDRAvg3Opt',
        in_channels=128,
        channels=128,
        num_classes=11,
        max_iters=iters,  # 🚀 【极其关键】必须把配置里的总步数传给你的调度器！
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.4),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                loss_weight=1.0),
            # dict(type='BoundaryLoss', loss_weight=20.0),
            dict(type='BoundaryLoss', loss_weight=1.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
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
    # dict(type='GenerateEdge', edge_width=4), # ⚠️ 跑你的方法时删掉这行
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(960, 720), keep_ratio=True),
    dict(type='Pad', size_divisor=32, pad_val=0), 
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# ================= 4. Dataloader 设置 =================
# ================= 4. Dataloader 设置 =================
train_dataloader = dict(
    batch_size=12,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        img_suffix='.png',       # ⚠️ 核心修复：告诉框架原图是 png
        seg_map_suffix='.png',   # ⚠️ 核心修复：告诉框架标签也是 png
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        img_suffix='.png',       # ⚠️ 加在这里
        seg_map_suffix='.png',   # ⚠️ 加在这里
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        img_suffix='.png',       # ⚠️ 加在这里
        seg_map_suffix='.png',   # ⚠️ 加在这里
        data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
        pipeline=test_pipeline))


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# ================= 5. 训练策略 =================


optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
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
