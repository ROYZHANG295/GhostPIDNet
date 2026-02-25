# 20260225:
# ROY: 知识蒸馏，基于 ghost-conv3-s 修改
# (可选但推荐) 调整学习率
# 因为你的模型已经有 75% mIoU 了，如果用初始的大学习率 (比如 0.01) 可能会震荡。
# 建议把初始学习率调小 2~5 倍。
# 把学习率从 0.005 降低到 0.0005（缩小 10 倍）

_base_ = [
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py'
]

# The class_weight is borrowed from https://github.com/openseg-group/OCNet.pytorch/issues/14 # noqa
# Licensed under the MIT License
class_weight = [
    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786,
    1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
    1.0507
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'  # noqa
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
# 改1：
# model = dict(
#     type='EncoderDecoder',
#     data_preprocessor=data_preprocessor,
#     backbone=dict(
#         type='PIDNetImprovedGhostConv',
#         in_channels=3,
#         channels=32,
#         ppm_channels=96,
#         num_stem_blocks=2,
#         num_branch_blocks=3,
#         align_corners=False,
#         norm_cfg=norm_cfg,
#         act_cfg=dict(type='ReLU', inplace=True),
#         init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
#     decode_head=dict(
#         type='PIDHead',
#         in_channels=128,
#         channels=128,
#         num_classes=19,
#         norm_cfg=norm_cfg,
#         act_cfg=dict(type='ReLU', inplace=True),
#         align_corners=True,
#         loss_decode=[
#             dict(
#                 type='CrossEntropyLoss',
#                 use_sigmoid=False,
#                 # class_weight=class_weight,
#                 loss_weight=0.4),
#             dict(
#                 type='OhemCrossEntropy',
#                 thres=0.9,
#                 min_kept=131072,
#                 class_weight=class_weight,
#                 loss_weight=1.0),
#             dict(type='BoundaryLoss', loss_weight=20.0),
#             dict(
#                 type='OhemCrossEntropy',
#                 thres=0.9,
#                 min_kept=131072,
#                 class_weight=class_weight,
#                 loss_weight=1.0)
#         ]),
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole'))

student_cfg = dict(
    type='EncoderDecoder',
    # 1. 定义学生的配置 (就是你原来的模型，但要把 data_preprocessor 删掉)
    # data_preprocessor=data_preprocessor,
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
#                class_weight=class_weight,
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

# ==================== 定义老师专用的 BN（防止 SyncBN 在 eval 时产生 NaN）====================
teacher_norm_cfg = dict(type='BN', requires_grad=False)

# ==================== 老师配置 (PIDNet-M) ====================
teacher_cfg = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='PIDNet',
        in_channels=3,
        channels=64,           # PIDNet-M 的基础通道数是 64
        ppm_channels=96,       # PIDNet-M 的 PPM 通道数是 96
        num_stem_blocks=2,     # PIDNet-M 的 stem blocks 数量是 2
        num_branch_blocks=3,   # PIDNet-M 的 branch blocks 数量是 3
        align_corners=False,
        norm_cfg=teacher_norm_cfg,  # 使用普通 BN，避免 SyncBN 的分布式同步问题
        act_cfg=dict(type='ReLU', inplace=True)),
    decode_head=dict(
        type='PIDHead',
        in_channels=256,       # PIDNet-M 的 Head 输入通道数是 256
        channels=256,          # PIDNet-M 的 Head 内部通道数是 256
        num_classes=19,
        norm_cfg=teacher_norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            dict(type='OhemCrossEntropy', thres=0.9, min_kept=131072, class_weight=class_weight, loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(type='OhemCrossEntropy', thres=0.9, min_kept=131072, class_weight=class_weight, loss_weight=1.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# 3. 组装最终的 model
model = dict(
    type='PIDNetDistillerWrapper',  
    student_cfg=student_cfg,
    teacher_cfg=teacher_cfg,
    # 【新增】在这里指定学生 12 万轮的预训练权重
    student_ckpt='checkpoints/pidnet-improved_ghost_conv_class_weight_I_only_s_2xb6-120k_1024x1024-cityscapes/iter_120000.pth',
    teacher_ckpt='checkpoints/pidnet-best/pidnet-m_2xb6-120k_1024x1024-cityscapes_20230301_143452-f9bcdbf3.pth',
    temperature=2.0,
    kd_weight=5.0,           # 语义蒸馏权重
    boundary_weight=5.0,     # 新增：边界蒸馏权重 (建议设为 5.0~10.0)
    data_preprocessor=data_preprocessor
)

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
# 改2：
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005)
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
