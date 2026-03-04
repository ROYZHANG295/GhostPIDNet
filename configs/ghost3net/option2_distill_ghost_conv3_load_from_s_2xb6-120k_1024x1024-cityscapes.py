_base_ = [
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py'
]

# ==================================================================
# 2. 训练超参设置
# ==================================================================
# 你的预训练权重路径 (77.8% 版本)
load_from = 'checkpoints/pidnet-improved_ghost_conv_class_weight_s_2xb6-120k_1024x1024-cityscapes/iter_240000-miou-77.8.pth'

resume = False  # 显式关闭 resume，因为我们要重置优化器状态
iters = 120000  # 再跑 120k iter

# 教师模型配置 (PIDNet-L) - 请确保路径正确
teacher_config_file = 'configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py'
teacher_checkpoint_file = 'work_dirs/pidnet-best/pidnet-l_2xb6-120k_1024x1024-cityscapes_20230303_114514-0783ca6b.pth'
# 如果你已经下载了 teacher 权重，请替换为本地路径，例如:
# teacher_checkpoint_file = 'checkpoints/pidnet_l_cityscapes.pth'

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

# ==================================================================
# 3. 模型定义 (集成蒸馏 Wrapper)
# ==================================================================
model = dict(
    type='EncoderDecoderKD',  # <--- 使用我们自定义的蒸馏类
    
    # --- Teacher 配置 ---
    teacher_config=teacher_config_file,
    teacher_ckpt=teacher_checkpoint_file,
    
    # --- 蒸馏 Loss 配置 ---
    distill_losses=[
        # A. Logits 蒸馏 (KL散度) - 学习软分类
        dict(type='KLDivergence',
             loss_name='loss_distill_logits',  # <--- 修改这里：把 name 改为 loss_name
             tau=1.0,
             loss_weight=3.0),
             
        # B. 特征蒸馏 (CWD) - 恢复纹理细节 (Wall/Fence)
        dict(type='ChannelWiseDivergence',
             loss_name='loss_distill_cwd',     # <--- 修改这里：把 name 改为 loss_name
             tau=1.0,
             loss_weight=5.0,
             # PIDNet-L (Teacher) I分支通道=256
             # GhostPIDNet-S (Student) I分支通道=128
             student_channels=128,
             teacher_channels=256)
    ],

    # --- Student 配置 (你的 GhostPIDNet) ---
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='PIDNetImprovedGhostConv', # 你的自定义 Backbone
        in_channels=3,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        # 这里的 init_cfg 在 load_from 生效时会被覆盖，可以保留
        init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth')
    ),
    decode_head=dict(
        type='PIDHead',
        in_channels=128,
        channels=128,
        num_classes=19,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        
        # --- 优化后的 Loss 组合 ---
        loss_decode=[
            # 1. Aux Loss (I分支)
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None, 
                loss_weight=0.4),

            # 2. Main Loss (主输出) - OHEM 开启
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=100000,
                class_weight=None, # 去掉 class_weight
                loss_weight=1.0),

            # 3. Boundary Loss (D分支) - 降权至 10.0
            dict(
                type='BoundaryLoss', 
                loss_weight=10.0), 

            # 4. Boundary-aware Semantic Loss
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=100000,
                class_weight=None,
                loss_weight=1.0)
        ]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ==================================================================
# 4. 数据流水线
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

# 注意：蒸馏比较吃显存。如果 OOM，请将 batch_size 调小 (如 4 或 2)
train_dataloader = dict(batch_size=6, dataset=dict(pipeline=train_pipeline))

# ==================================================================
# 5. 优化器与调度策略
# ==================================================================
# 学习率调整：从 0.01 降到 0.005，因为是基于 77.8% 的模型微调
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

# ==================================================================
# 6. 运行逻辑
# ==================================================================
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=iters, val_interval=iters // 120)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=iters // 20),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

randomness = dict(seed=304)

# 测试 Pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='Pad', size_divisor=32, pad_val=0), 
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
