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
    🏆 HALO: Holistic Asymmetric Laplacian Oracle
    ===========================================================================
    【完美消融实验版 - 严格对齐官方 DDRNet Baseline】
    1. Baseline: 仅保留 Context 和 Spatial 语义分支 (loss 为 0)
    2. use_laplacian: 开启边界头，使用 Laplacian GT + BCE + Dice 联合监督
    3. use_halo_feedback: 开启跨分支先知反哺 (Topology-Aware Injection)
    ===========================================================================
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 max_iters: int = 120000,  
                 # =========================================================
                 # 🔬 消融实验控制开关
                 # =========================================================
                 use_laplacian: bool = True,           # 开启: 拉普拉斯边界提取 + 联合 Dice 损失
                 use_dynamic_dilation: bool = True,    # 开启: 动态形态学膨胀
                 use_3stage_schedule: bool = True,     # 开启: 三阶段动态权重调度
                 use_halo_feedback: bool = True,       # 开启: 跨分支先知反哺
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
        
        self.use_laplacian = use_laplacian
        self.use_dynamic_dilation = use_dynamic_dilation
        self.use_3stage_schedule = use_3stage_schedule
        self.use_halo_feedback = use_halo_feedback
        
        self.register_buffer('local_step', torch.tensor(0, dtype=torch.long))
        
        # 【HALO 新增】：独立的单通道边界预测头
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
            
            # Context 分支 (语义)
            x_c = self.head(c5_feat)
            x_c_logit = self.cls_seg(x_c)
            
            # Spatial 分支 (语义)
            x_s = self.aux_head(c3_feat)
            x_s_logit = self.aux_cls_seg(x_s)
            
            # Spatial 分支附带的边界提取器 (HALO 新增)
            bd_logit = self.bd_cls_seg(x_s)

            return x_c_logit, x_s_logit, bd_logit
        else:
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
            0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
            t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
            t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
            t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
            t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.1},
            max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.1}
        }
        return schedule

    def _get_dynamic_params(self, current_step: int) -> Tuple[int, float, float]:
        schedule = self.dynamic_schedule
        milestones = sorted(schedule.keys())
        
        if current_step <= milestones[0]: 
            cfg = schedule[milestones[0]]
            cur_dilation, cur_dice_w, cur_fb_w = cfg['dilation'], cfg['dice_w'], cfg['fb_w']
        elif current_step >= milestones[-1]: 
            cfg = schedule[milestones[-1]]
            cur_dilation, cur_dice_w, cur_fb_w = cfg['dilation'], cfg['dice_w'], cfg['fb_w']
        else:
            start_step, end_step = milestones[0], milestones[-1]
            for i in range(len(milestones) - 1):
                if milestones[i] <= current_step < milestones[i+1]:
                    start_step, end_step = milestones[i], milestones[i+1]
                    break
                    
            start_cfg, end_cfg = schedule[start_step], schedule[end_step]
            progress = (current_step - start_step) / float(end_step - start_step)
            
            cur_dilation = start_cfg['dilation']
            cur_dice_w = start_cfg['dice_w'] + progress * (end_cfg['dice_w'] - start_cfg['dice_w'])
            cur_fb_w = start_cfg['fb_w'] + progress * (end_cfg['fb_w'] - start_cfg['fb_w'])
            
        if not self.use_dynamic_dilation:
            cur_dilation = 3  
            
        if not self.use_3stage_schedule:
            cur_dice_w = 1.0  
            cur_fb_w = 1.0    
            
        return cur_dilation, cur_dice_w, cur_fb_w

    def loss_by_feat(self, seg_logits: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        
        if self.training:
            self.local_step += 1
        current_step = self.local_step.item()
        
        cur_dilation, cur_dice_w, cur_fb_w = self._get_dynamic_params(current_step)

        loss = dict()
        context_logit, spatial_logit, bd_logit = seg_logits
        seg_label = self._stack_batch_gt(batch_data_samples)

        context_logit = resize(context_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        spatial_logit = resize(spatial_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        bd_logit = resize(bd_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        
        seg_label = seg_label.squeeze(1)

        # 1. 官方原版 Baseline Loss (Context 语义 + Spatial 语义)
        loss['loss_context'] = self.loss_decode[0](context_logit, seg_label)
        loss['loss_spatial'] = self.loss_decode[1](spatial_logit, seg_label)
        
        # =====================================================================
        # 🔬 消融点 1：Laplacian Boundary Module (BCE + Dice)
        # =====================================================================
        if self.use_laplacian:
            # 开启时：计算真实的边界 Loss
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
        else:
            # 关闭时：退化为 0 (乘以 0.0 防止 DDP 报错，完美等价于原版 Baseline)
            loss['loss_bd_laplacian'] = 0.0 * bd_logit.sum()
        
        # =====================================================================
        # 🔬 消融点 4：Topology-Aware Injection (Cross-Branch Oracle Feedback)
        # =====================================================================
        if self.use_halo_feedback:
            # 只有在开启反哺时，才计算并砸向 Context 分支
            pred_sigmoid = torch.sigmoid(bd_logit[:, 0, :, :])
            bd_pred_mask = (pred_sigmoid > 0.5)
            filler = torch.ones_like(seg_label) * self.ignore_index
            halo_label = torch.where(bd_pred_mask, seg_label, filler)
            
            loss['loss_halo_feedback'] = self.loss_decode[0](context_logit, halo_label) * cur_fb_w
        else:
            # 关闭时：退化为 0
            loss['loss_halo_feedback'] = 0.0 * context_logit.sum()
            
        loss['acc_seg'] = accuracy(context_logit, seg_label, ignore_index=self.ignore_index)

        return loss
