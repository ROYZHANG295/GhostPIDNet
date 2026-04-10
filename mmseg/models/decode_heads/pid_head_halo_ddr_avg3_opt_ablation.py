# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F  
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList


class BasePIDHead(BaseModule):
    """Base class for PID head."""
    
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.conv = ConvModule(
            in_channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            order=('norm', 'act', 'conv')
        )
        _, self.norm = build_norm_layer(norm_cfg, num_features=channels)
        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor, cls_seg: Optional[nn.Module]) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if cls_seg is not None:
            x = cls_seg(x)
        return x


@MODELS.register_module()
class PIDHeadHALOSameDDRAvg3OptAblation(BaseDecodeHead):
    """
    ===========================================================================
    🏆 HALO: Topology-Aware Boundary Supervision (教科书级消融版)
    ===========================================================================
    变量控制说明：
    - use_olb (On-the-fly Laplacian Boundary): 是否开启辅助分支的边界监督
    - use_dd  (Dynamic Dilation): 是否开启动态膨胀 (若关闭则固定 Dilation=3)
    - use_fb  (Cross-Branch Oracle Feedback): 是否开启跨分支先知反哺
    - use_aas (Architecture-Aware Scheduling): 是否开启解耦架构感知调度
    ===========================================================================
    """
    
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 max_iters: int = 120000, 
                 
                 # ================== 🔬 核心消融开关 ==================
                 use_olb: bool = True,  # 1. 基础拉普拉斯边界
                 use_dd: bool = True,   # 2. 动态膨胀策略
                 use_fb: bool = True,   # 3. 跨分支先知反哺
                 use_aas: bool = True,  # 4. 架构感知解耦调度
                 # ====================================================
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs
        )
        
        self.register_buffer('local_step', torch.tensor(0, dtype=torch.long))
        self.max_iters = max_iters
        
        # 保存消融变量
        self.use_olb = use_olb
        self.use_dd = use_dd
        self.use_fb = use_fb
        self.use_aas = use_aas

        self.i_head = BasePIDHead(in_channels, channels, norm_cfg, act_cfg)
        self.p_head = BasePIDHead(in_channels // 2, channels, norm_cfg, act_cfg)
        self.d_head = BasePIDHead(in_channels // 2, in_channels // 4, norm_cfg)
        
        self.p_cls_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        self.d_cls_seg = nn.Conv2d(in_channels // 4, 1, kernel_size=1)

        # 构建调度表
        self.dynamic_schedule = self._build_avg3_schedule(self.max_iters)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs: Union[Tensor, Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        if self.training:
            x_p, x_i, x_d = inputs
            x_p = self.p_head(x_p, self.p_cls_seg)
            x_i = self.i_head(x_i, self.cls_seg)
            x_d = self.d_head(x_d, self.d_cls_seg)
            return x_p, x_i, x_d
        else:
            return self.i_head(inputs, self.cls_seg)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def _generate_laplacian_boundary(self, 
                                     semantic_gt: Tensor, 
                                     num_classes: int, 
                                     ignore_index: int = 255, 
                                     dilation_size: int = 3) -> Tensor:
        
        valid_mask = (semantic_gt != ignore_index).float().unsqueeze(1)
        clean_gt = torch.where(
            semantic_gt == ignore_index, 
            torch.zeros_like(semantic_gt), 
            semantic_gt
        )
        
        gt_onehot = F.one_hot(clean_gt, num_classes=num_classes)
        gt_onehot = gt_onehot.permute(0, 3, 1, 2).float()
        
        laplacian_kernel = torch.tensor([
            [1.0,  1.0, 1.0],
            [1.0, -8.0, 1.0],
            [1.0,  1.0, 1.0]
        ], device=semantic_gt.device, dtype=torch.float32).view(1, 1, 3, 3)
        
        laplacian_kernel = laplacian_kernel.repeat(num_classes, 1, 1, 1)
        
        edge = F.conv2d(gt_onehot, laplacian_kernel, padding=1, groups=num_classes)
        edge = (torch.abs(edge) > 0.1).float()
        boundary_map = torch.max(edge, dim=1, keepdim=True)[0]
        
        if dilation_size > 1:
            pad = dilation_size // 2
            boundary_map = F.max_pool2d(
                boundary_map, 
                kernel_size=dilation_size, 
                stride=1, 
                padding=pad
            )
            boundary_map = boundary_map[:, :, :valid_mask.shape[2], :valid_mask.shape[3]]
            
        boundary_map = boundary_map * valid_mask
        return boundary_map.squeeze(1)

    def _build_avg3_schedule(self, max_iters: int) -> dict:
        t1 = int(max_iters / 3.0)
        t2 = int(max_iters * 2.0 / 3.0)

        if self.use_aas:
            # [AAS 开启] 解耦稳压调度：激进的内部边界权重，恒定的外部反哺保护
            return {
                0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
                t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
                t1 + 1:    {'dilation': 4, 'dice_w': 1.0, 'fb_w': 1.0},
                t2:        {'dilation': 4, 'dice_w': 1.0, 'fb_w': 1.0},
                t2 + 1:    {'dilation': 3, 'dice_w': 0.5, 'fb_w': 1.0},
                max_iters: {'dilation': 3, 'dice_w': 0.5, 'fb_w': 1.0}
            }
        else:
            # [AAS 关闭] 传统固定调度：所有权重均为 1.0
            return {
                0:         {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
                t1:        {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
                t1 + 1:    {'dilation': 4, 'dice_w': 1.0, 'fb_w': 1.0},
                t2:        {'dilation': 4, 'dice_w': 1.0, 'fb_w': 1.0},
                t2 + 1:    {'dilation': 3, 'dice_w': 1.0, 'fb_w': 1.0},
                max_iters: {'dilation': 3, 'dice_w': 1.0, 'fb_w': 1.0}
            }

    def _get_dynamic_params(self, current_step: int) -> Tuple[int, float, float]:
        schedule = self.dynamic_schedule
        milestones = sorted(schedule.keys())
        
        start_step = milestones[0]
        for i in range(len(milestones) - 1):
            if milestones[i] <= current_step <= milestones[i+1]:
                start_step = milestones[i]
                break
                
        cfg = schedule[start_step]
        
        # 控制 DD (Dynamic Dilation) 消融
        cur_dilation = cfg['dilation'] if self.use_dd else 3
            
        return cur_dilation, cfg['dice_w'], cfg['fb_w']

    def loss_by_feat(self, seg_logits: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        
        if self.training:
            self.local_step += 1
        current_step = self.local_step.item()
        
        cur_dilation, cur_dice_w, cur_fb_w = self._get_dynamic_params(current_step)

        loss = dict()
        p_logit, i_logit, d_logit = seg_logits
        sem_label = self._stack_batch_gt(batch_data_samples)

        p_logit = resize(input=p_logit, size=sem_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        i_logit = resize(input=i_logit, size=sem_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        d_logit = resize(input=d_logit, size=sem_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        
        sem_label = sem_label.squeeze(1)

        bd_label = self._generate_laplacian_boundary(
            sem_label, num_classes=self.num_classes, ignore_index=self.ignore_index, dilation_size=cur_dilation
        )

        # 1. 基础全局语义损失
        loss['loss_sem_p'] = self.loss_decode[0](p_logit, sem_label, ignore_index=self.ignore_index)
        loss['loss_sem_i'] = self.loss_decode[1](i_logit, sem_label)
        
        # 先提取 D 分支预测概率
        pred_sigmoid = torch.sigmoid(d_logit[:, 0, :, :])
        
        # 2. 联合边界损失 (控制 OLB 消融)
        if self.use_olb:
            bce_loss = self.loss_decode[2](d_logit, bd_label)
            valid_mask = (sem_label != self.ignore_index).float()
            intersection = (pred_sigmoid * bd_label * valid_mask).sum(dim=(1, 2))
            union = (pred_sigmoid * valid_mask).sum(dim=(1, 2)) + (bd_label * valid_mask).sum(dim=(1, 2))
            dice_loss = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
            
            loss['loss_bd_laplacian'] = bce_loss + cur_dice_w * dice_loss
        else:
            loss['loss_bd_laplacian'] = d_logit.sum() * 0.0

        # 3. 跨分支先知反哺 (控制 FB 消融)
        if self.use_fb:
            bd_pred_mask = (pred_sigmoid > 0.5)
            filler = torch.ones_like(sem_label) * self.ignore_index
            halo_label = torch.where(bd_pred_mask, sem_label, filler)
                
            loss['loss_halo_feedback'] = self.loss_decode[3](i_logit, halo_label) * cur_fb_w
        else:
            loss['loss_halo_feedback'] = i_logit.sum() * 0.0

        # 4. 准确率统计
        loss['acc_seg'] = accuracy(i_logit, sem_label, ignore_index=self.ignore_index)
            
        return loss
