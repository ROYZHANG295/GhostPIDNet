_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    
    # =========================================================
    # 1. 替换为我们自定义的“暴露细节分支”的 Backbone
    # =========================================================
    backbone=dict(
        type='ExposedBiSeNetV2',
        # 索引说明: 
        # 0: x_head (BGA融合特征)
        # 1~4: 语义分支中间特征 (留给 auxiliary_head 用)
        # 5: x_detail (细节分支特征，我们的拉普拉斯算子专用)
        out_indices=(0, 1, 2, 3, 4, 5) 
    ),
    
    # =========================================================
    # 2. 替换为我们的 HALO 动态拉普拉斯头
    # =========================================================
    decode_head=dict(
        type='BiSeNetHALOHead',  
        in_channels=[128, 128],  # [Detail通道数, Fused通道数]
        in_index=[5, 0],         # 精准抓取: inputs[0] = detail(5), inputs[1] = fused(0)
        max_iters=160000,        # 告诉 HALO 您要跑多少步，它会自动切分黄金比例
        # 显式声明关键参数，防止被 base config 覆盖
        channels=1024,
        num_classes=19,
        align_corners=False,
        
        # 这里的 loss_decode 可以不用写，因为我们在 Head 内部重写了 loss_by_feat
    )
    # 注意: _base_ 里的 auxiliary_head 会自动继承并挂载在索引 1, 2, 3, 4 上，完美兼容！
)

param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=160000,
        by_epoch=False,
    )
]
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
