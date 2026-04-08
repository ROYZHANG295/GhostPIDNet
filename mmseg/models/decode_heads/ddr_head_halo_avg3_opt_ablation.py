# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList


@MODELS.register_module()
class DDRHeadHALOAvg3OptAblation(BaseDecodeHead):
    """
    ===========================================================================
    🏆 HALO: Holistic Asymmetric Laplacian Oracle (全局非对称拉普拉斯先知)
    ===========================================================================
    
    【Table 3 消融实验 (Ablation Study) 终极配置指南】
    通过修改以下参数，可一键复现论文中的所有消融节点与崩溃曲线：

    [1] Baseline (原版 DDRNet):
        use_boundary = False
        (其余参数随意，因为根本不走边界计算分支)

    [2] + Online Laplacian Boundary:
        use_boundary = True
        use_dynamic_dilation = False (使用固定粗细的边界)
        static_dilation_val = 3      (固定宽度为 3)
        scheduler_type = 'constant'  (权重死锁 1.0)
        use_oracle_feedback = False  (不反哺，仅监督 Spatial 分支)

    [3] + Dynamic Dilation (证明 5->4->3 优于固定 3):
        use_boundary = True
        use_dynamic_dilation = True  (开启 5->4->3 动态膨胀)
        scheduler_type = 'constant'  (权重依然死锁 1.0)
        use_oracle_feedback = False

    [4] + Piecewise-Constant Scheduler (证明硬跳变优于死锁 1.0):
        use_boundary = True
        use_dynamic_dilation = True
        scheduler_type = 'piecewise' (开启 1.0 -> 0.5 -> 0.1 阶梯降权)
        use_oracle_feedback = False

    [5] + Topology-Aware Injection (Full HALO 最终绝杀版):
        use_boundary = True
        use_dynamic_dilation = True
        scheduler_type = 'piecewise'
        use_oracle_feedback = True   (开启跨分支先知反哺，将边界知识砸向 Context 分支)

    [额外] Figure 8 灾难崩溃曲线 (反面教材):
        scheduler_type = 'smooth'    (开启平滑衰减，复现 mIoU 断崖式暴跌)
    ===========================================================================
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 max_iters: int = 120000,
                 # --- 消融实验核心配置开关 ---
                 use_boundary: bool = True,
                 use_dynamic_dilation: bool = True,
                 static_dilation_val: int = 3,       # <--- 关闭动态膨胀时的固定基准值
                 scheduler_type: str = 'piecewise',  # 'constant', 'smooth', 'piecewise'
                 use_oracle_feedback: bool = True,
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.head = self._make_base_head(self.in_channels, self.channels)
        self.aux_head = self._make_base_head(self.in_channels // 2, self.channels)
        self.aux_cls_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)
        
        self.max_iters = max_iters
        
        # --- 记录消融标志位 ---
        self.use_boundary = use_boundary
        self.use_dynamic_dilation = use_dynamic_dilation
        self.static_dilation_val = static_dilation_val
        self.scheduler_type = scheduler_type
        self.use_oracle_feedback = use_oracle_feedback
        
        self.register_buffer('local_step', torch.tensor(0, dtype=torch.long))
        
        # 仅在开启边界时初始化边界预测头和调度器，保证 Baseline 绝对干净
        if self.use_boundary:
            self.bd_cls_seg = nn.Conv2d(self.channels, 1, kernel_size=1)
            self.dynamic_schedule = self._build_avg3_schedule(self.max_iters)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_base_head(self, in_channels: int, channels: int) -> nn.Sequential:
        layers = [
            ConvModule(
                in_channels, channels, kernel_size=3, padding=1,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
                order=('norm', 'act', 'conv')),
            build_norm_layer(self.norm_cfg, channels)[1],
            build_activation_layer(self.act_cfg),
        ]
        return nn.Sequential(*layers)

    def forward(self, inputs: Union[Tensor, Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        if self.training:
            c3_feat, c5_feat = inputs
            
            x_c = self.head(c5_feat)
            x_c_logit = self.cls_seg(x_c)
            
            x_s = self.aux_head(c3_feat)
            x_s_logit = self.aux_cls_seg(x_s)
            
            bd_logit = None
            if self.use_boundary:
                bd_logit = self.bd_cls_seg(x_s)

            return x_c_logit, x_s_logit, bd_logit
        else:
            # 推理阶段：完全丢弃高分辨率分支和边界计算，0 FLOPs 增加！
            x_c = self.head(inputs)
            x_c = self.cls_seg(x_c)
            return x_c

    def _generate_laplacian_boundary(self, semantic_gt: Tensor, num_classes: int, 
                                     ignore_index: int = 255, dilation_size: int = 3) -> Tensor:
        valid_mask = (semantic_gt != ignore_index).float().unsqueeze(1)
        clean_gt = torch.where(semantic_gt == ignore_index, torch.zeros_like(semantic_gt), semantic_gt)
        gt_onehot = F.one_hot(clean_gt, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        laplacian_kernel = torch.tensor([
            [1.0,  1.0, 1.0],
            [1.0, -8.0, 1.0],
            [1.0,  1.0, 1.0]
        ], device=semantic_gt.device, dtype=torch.float32).view(1, 1, 3, 3).repeat(num_classes, 1, 1, 1)
        
        edge = F.conv2d(gt_onehot, laplacian_kernel, padding=1, groups=num_classes)
        edge = (torch.abs(edge) > 0.1).float()
        boundary_map = torch.max(edge, dim=1, keepdim=True)[0]
        
        if dilation_size > 1:
            pad = dilation_size // 2
            boundary_map = F.max_pool2d(boundary_map, kernel_size=dilation_size, stride=1, padding=pad)
            boundary_map = boundary_map[:, :, :valid_mask.shape[2], :valid_mask.shape[3]]
            
        return (boundary_map * valid_mask).squeeze(1)

    def _build_avg3_schedule(self, max_iters: int) -> dict:
        t1 = int(max_iters / 3.0)
        t2 = int(max_iters * 2.0 / 3.0)

        schedule = {
            0:         {'dilation': 5, 'dice_w': 1.0},
            t1:        {'dilation': 5, 'dice_w': 1.0},
            t1 + 1:    {'dilation': 4, 'dice_w': 0.5},
            t2:        {'dilation': 4, 'dice_w': 0.5},
            t2 + 1:    {'dilation': 3, 'dice_w': 0.1},
            max_iters: {'dilation': 3, 'dice_w': 0.1}
        }
        return schedule

    def _get_dynamic_params(self, current_step: int) -> Tuple[int, float]:
        schedule = self.dynamic_schedule
        milestones = sorted(schedule.keys())
        
        # 1. 定位当前所处的阶段
        start_step, end_step = milestones[0], milestones[-1]
        for i in range(len(milestones) - 1):
            if milestones[i] <= current_step <= milestones[i+1]:
                start_step, end_step = milestones[i], milestones[i+1]
                break
                
        start_cfg, end_cfg = schedule[start_step], schedule[end_step]
        
        # =====================================================================
        # 2. 处理 Dilation (消融开关：动态 5->4->3 vs 固定 3)
        # =====================================================================
        if self.use_dynamic_dilation:
            cur_dilation = start_cfg['dilation']
        else:
            cur_dilation = self.static_dilation_val  # 默认使用固定宽度 3
            
        # =====================================================================
        # 3. 处理 Weight Scheduler (消融开关：恒定 vs 平滑 vs 阶梯硬跳变)
        # =====================================================================
        if self.scheduler_type == 'constant':
            cur_dice_w = 1.0  # 永远保持最强监督 1.0
            
        elif self.scheduler_type == 'piecewise':
            cur_dice_w = start_cfg['dice_w']  # 阶梯硬跳变 1.0 -> 0.5 -> 0.1，锁定不插值
            
        elif self.scheduler_type == 'smooth':
            # 平滑衰减 (Smooth Decay)：产生 Moving Target，导致灾难性崩溃
            if end_step == start_step:
                progress = 0.0
            else:
                progress = (current_step - start_step) / float(end_step - start_step)
            cur_dice_w = start_cfg['dice_w'] + progress * (end_cfg['dice_w'] - start_cfg['dice_w'])
            
        else:
            raise ValueError(f"Unknown scheduler_type: {self.scheduler_type}")
            
        return cur_dilation, cur_dice_w

    def loss_by_feat(self, seg_logits: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        
        if self.training:
            self.local_step += 1
        current_step = self.local_step.item()

        loss = dict()
        context_logit, spatial_logit, bd_logit = seg_logits
        seg_label = self._stack_batch_gt(batch_data_samples)

        context_logit = resize(context_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        spatial_logit = resize(spatial_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        if bd_logit is not None:
            bd_logit = resize(bd_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        
        seg_label = seg_label.squeeze(1)

        # =====================================================================
        # 步骤 1: 基础全局语义损失 (Baseline)
        # =====================================================================
        loss['loss_context'] = self.loss_decode[0](context_logit, seg_label)
        loss['loss_spatial'] = self.loss_decode[1](spatial_logit, seg_label)
        loss['acc_seg'] = accuracy(context_logit, seg_label, ignore_index=self.ignore_index)
        
        # 如果未开启边界辅助，直接返回 Baseline 的 loss (消融节点 1)
        if not self.use_boundary:
            return loss

        # =====================================================================
        # 步骤 2: 拉普拉斯边界提取与联合损失计算
        # =====================================================================
        cur_dilation, cur_dice_w = self._get_dynamic_params(current_step)

        bd_label = self._generate_laplacian_boundary(
            seg_label, num_classes=self.num_classes, ignore_index=self.ignore_index, dilation_size=cur_dilation
        )

        bce_loss = F.binary_cross_entropy_with_logits(bd_logit.squeeze(1), bd_label)
        
        pred_sigmoid = torch.sigmoid(bd_logit[:, 0, :, :])
        valid_mask = (seg_label != self.ignore_index).float()
        intersection = (pred_sigmoid * bd_label * valid_mask).sum(dim=(1, 2))
        union = (pred_sigmoid * valid_mask).sum(dim=(1, 2)) + (bd_label * valid_mask).sum(dim=(1, 2))
        dice_loss = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
        
        loss['loss_bd_laplacian'] = bce_loss + cur_dice_w * dice_loss
        
        # =====================================================================
        # 步骤 3: 跨分支先知反哺 (Topology-Aware Injection)
        # =====================================================================
        if self.use_oracle_feedback:
            # 提取 Spatial 分支 (先知) 预测的边界掩码
            bd_pred_mask = (pred_sigmoid > 0.5)
            
            # 制作“纯边界语义标签”：只保留先知认为是边界的地方，其余忽略
            filler = torch.ones_like(seg_label) * self.ignore_index
            halo_label = torch.where(bd_pred_mask, seg_label, filler)
            
            # 将严苛的边界标签作为强监督信号砸向 Context 分支！
            loss['loss_halo_feedback'] = self.loss_decode[0](context_logit, halo_label) * cur_dice_w

        return loss
