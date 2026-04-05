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
class PIDHeadLaplacianOpt3Dynamic6SmoothSRA(BaseDecodeHead):
    
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs
        )
        
        # [工程实现]: 注册一个内置的计步器，随模型一起保存在 checkpoint 中，支持断点续训
        self.register_buffer('local_step', torch.tensor(0, dtype=torch.long))

        self.i_head = BasePIDHead(in_channels, channels, norm_cfg, act_cfg)
        self.p_head = BasePIDHead(in_channels // 2, channels, norm_cfg, act_cfg)
        self.d_head = BasePIDHead(in_channels // 2, in_channels // 4, norm_cfg)
        
        self.p_cls_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        self.d_cls_seg = nn.Conv2d(in_channels // 4, 1, kernel_size=1)

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

    # =========================================================================
    # [创新点 1]: 在线拉普拉斯边界提取 (On-the-fly Laplacian Boundary Extraction)
    # =========================================================================
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
        
        # =====================================================================
        # [创新点 2]: 动态形态学膨胀 (Dynamic Morphological Dilation)
        # =====================================================================
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

    # =========================================================================
    # [创新点 3]: 平滑动态课程学习调度器 (Smooth Dynamic Curriculum Scheduler)
    # [修改说明]: 彻底移除了 thresh 参数！因为采用了连续软掩码，模型实现了免调参！
    # =========================================================================
    def _get_dynamic_params(self, current_step: int) -> Tuple[int, float]:
        
        schedule = {
            0:      {'step': {'dilation': 5}, 'smooth': {'dice_w': 3.0}},
            40000:  {'step': {'dilation': 5}, 'smooth': {'dice_w': 3.0}},
            80000:  {'step': {'dilation': 5}, 'smooth': {'dice_w': 1.0}},
            80001:  {'step': {'dilation': 3}, 'smooth': {'dice_w': 0.5}},
            120000: {'step': {'dilation': 3}, 'smooth': {'dice_w': 0.5}} 
        }

        milestones = sorted(schedule.keys())
        
        if current_step <= milestones[0]:
            cfg = schedule[milestones[0]]
            return cfg['step']['dilation'], cfg['smooth']['dice_w']
            
        if current_step >= milestones[-1]:
            cfg = schedule[milestones[-1]]
            return cfg['step']['dilation'], cfg['smooth']['dice_w']
            
        start_step, end_step = milestones[0], milestones[-1]
        for i in range(len(milestones) - 1):
            if milestones[i] <= current_step < milestones[i+1]:
                start_step, end_step = milestones[i], milestones[i+1]
                break
                
        start_cfg, end_cfg = schedule[start_step], schedule[end_step]
        progress = (current_step - start_step) / float(end_step - start_step)
        
        cur_dilation = start_cfg['step']['dilation']
        cur_dice_w = start_cfg['smooth']['dice_w'] + progress * (
            end_cfg['smooth']['dice_w'] - start_cfg['smooth']['dice_w']
        )
        
        return cur_dilation, cur_dice_w

    def loss_by_feat(self, seg_logits: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        
        if self.training:
            self.local_step += 1
        current_step = self.local_step.item()
        
        # 获取当前步数下的平滑参数 (不再需要 thresh)
        cur_dilation, cur_dice_w = self._get_dynamic_params(current_step)

        loss = dict()
        p_logit, i_logit, d_logit = seg_logits
        sem_label = self._stack_batch_gt(batch_data_samples)

        p_logit = resize(input=p_logit, size=sem_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        i_logit = resize(input=i_logit, size=sem_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        d_logit = resize(input=d_logit, size=sem_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        
        sem_label = sem_label.squeeze(1)

        bd_label = self._generate_laplacian_boundary(
            sem_label, 
            num_classes=self.num_classes, 
            ignore_index=self.ignore_index, 
            dilation_size=cur_dilation
        )

        # 1. 语义主干 Loss
        loss['loss_sem_p'] = self.loss_decode[0](p_logit, sem_label, ignore_index=self.ignore_index)
        loss['loss_sem_i'] = self.loss_decode[1](i_logit, sem_label)
        
        # 2. 拓扑联合边界 Loss
        bce_loss = self.loss_decode[2](d_logit, bd_label)
        pred_sigmoid = torch.sigmoid(d_logit[:, 0, :, :])
        
        valid_mask = (sem_label != self.ignore_index).float()
        intersection = (pred_sigmoid * bd_label * valid_mask).sum(dim=(1, 2))
        union = (pred_sigmoid * valid_mask).sum(dim=(1, 2)) + (bd_label * valid_mask).sum(dim=(1, 2))
        dice_loss = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
        
        loss['loss_bd_laplacian'] = bce_loss + cur_dice_w * dice_loss
        
        # =====================================================================
        # [创新点 4]: 连续软掩码反哺机制 (Continuous Soft-Mask Feedback)
        # [为什么]: 原版的硬阈值 (如 pred > 0.8) 在小 Batch Size 下极易受到 BN 抖动影响，
        #          导致边界像素被反复误杀，引发严重的“梯度截断”和训练崩溃。
        # [效果]: 彻底抛弃硬阈值！将边界概率图 (0~1) 扩展为空间注意力 (Spatial Attention)，
        #        以残差相乘的方式直接注入语义 Logit。不仅实现了免调参 (Parameter-free)，
        #        而且 OHEM 损失会自动在注意力高的地方挖掘困难样本，完美抵御 BN 抖动！
        # =====================================================================
        
        # 将概率图 [B, H, W] 扩展为 [B, 1, H, W]，以便利用广播机制与 i_logit 相乘
        boundary_attention = pred_sigmoid.unsqueeze(1) 
        
        # 残差特征增强 (Residual Feature Enhancement)
        # 如果是边界 (attention -> 1)，Logit 被放大 2 倍，Loss 惩罚加剧；
        # 如果不是边界 (attention -> 0)，Logit 保持原样 (乘 1.0)。
        enhanced_i_logit = i_logit * (1.0 + boundary_attention)
        
        # 直接在全图上计算增强后的 Loss，不再使用 ignore_index 截断！
        loss['loss_sem_bd'] = self.loss_decode[3](enhanced_i_logit, sem_label) 
        
        # 3. 准确率统计
        loss['acc_seg'] = accuracy(i_logit, sem_label, ignore_index=self.ignore_index)
            
        return loss
