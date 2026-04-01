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
class PIDHeadHALO(BaseDecodeHead):
    """
    ===========================================================================
    🏆 HALO: Holistic Asymmetric Laplacian Oracle (PID 架构版)
    ===========================================================================
    【核心创新整合】：
    
    1. 零开销拉普拉斯先知 (Zero-Cost Laplacian Oracle):
       直接从 Semantic GT 中利用拉普拉斯算子提取边缘，确保 100% 绝对对齐，
       获得极其纯粹、锐利的高频边界特征，且推理时 0 开销。
       
    2. 动态黄金分割课程学习 (Holistic Golden Ratio Curriculum):
       【彻底消灭梯度休克与后期震荡】摒弃硬编码，引入 0.382-0.236-0.382 黄金调度！
       - Phase 1 (0~38.2%): 强先验期 (Dilation=5, Weight=3.0)。
       - Phase 2 (38.2%~61.8%): 平滑过渡期 (Dilation=4, Weight 线性插值)。
       - Phase 3 (61.8%~100%): 极速冲刺期 (Dilation=3, Weight=0.5)。
         在极低学习率下提供长达 38.2% 的无突变微调期，彻底抚平震荡，冲击 SOTA！
         
    3. 神谕直连语义反哺 (Oracle-Direct Semantic Feedback):
       【重大重构】彻底抛弃原版极不稳定的自预测置信度阈值 (pred > thresh)，
       直接将绝对对齐的拉普拉斯物理边界作为强监督掩码反哺给语义主干。
       从数学上切断了由于权重衰减引发的负反馈循环，实现全周期的绝对稳定！
    ===========================================================================
    """
    
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 max_iters: int = 120000,  # <--- 传入总训练步数，默认 120k
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs
        )
        
        # [工程实现]: 注册内置计步器，随模型一起保存在 checkpoint 中，支持断点续训
        self.register_buffer('local_step', torch.tensor(0, dtype=torch.long))
        self.max_iters = max_iters

        self.i_head = BasePIDHead(in_channels, channels, norm_cfg, act_cfg)
        self.p_head = BasePIDHead(in_channels // 2, channels, norm_cfg, act_cfg)
        self.d_head = BasePIDHead(in_channels // 2, in_channels // 4, norm_cfg)
        
        self.p_cls_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        self.d_cls_seg = nn.Conv2d(in_channels // 4, 1, kernel_size=1)

        # =====================================================================
        # 【HALO 核心引擎】：初始化时自动计算黄金分割调度表
        # =====================================================================
        self.dynamic_schedule = self._build_golden_schedule(self.max_iters)

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
        
        # [创新点 2]: 动态形态学膨胀 (由粗到细 Coarse-to-Fine)
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
    # [创新点 3]: HALO 黄金分割课程学习调度器 (Holistic Golden Ratio Curriculum)
    # =========================================================================
    def _build_golden_schedule(self, max_iters: int) -> dict:
        t1 = int(max_iters * 0.382)  # 黄金分割点 1
        t2 = int(max_iters * 0.618)  # 黄金分割点 2
        
        # 【修改点 1】：彻底移除极度危险的 thresh 参数，切断负反馈循环
        schedule = {
            # 阶段 1 (0 -> 38.2%)：强先验平稳期。给网络下“猛药”。
            0:      {'dilation': 5, 'dice_w': 3.0},
            t1:     {'dilation': 5, 'dice_w': 3.0},
            
            # 阶段 2 (38.2% -> 61.8%)：平滑过渡期。加入 Dilation=4 缓冲！
            t1 + 1: {'dilation': 4, 'dice_w': 3.0},
            t2:     {'dilation': 4, 'dice_w': 1.0},
            
            # 阶段 3 (61.8% -> 100%)：瞬间减负与精细微调期。复刻一飞冲天的奇迹！
            t2 + 1: {'dilation': 3, 'dice_w': 0.5},
            max_iters: {'dilation': 3, 'dice_w': 0.5} 
        }
        return schedule

    def _get_dynamic_params(self, current_step: int) -> Tuple[int, float]:
        schedule = self.dynamic_schedule
        milestones = sorted(schedule.keys())
        
        # 【修改点 2】：适配移除 thresh 后的读取逻辑
        if current_step <= milestones[0]:
            cfg = schedule[milestones[0]]
            return cfg['dilation'], cfg['dice_w']
            
        if current_step >= milestones[-1]:
            cfg = schedule[milestones[-1]]
            return cfg['dilation'], cfg['dice_w']
            
        start_step, end_step = milestones[0], milestones[-1]
        for i in range(len(milestones) - 1):
            if milestones[i] <= current_step < milestones[i+1]:
                start_step, end_step = milestones[i], milestones[i+1]
                break
                
        start_cfg, end_cfg = schedule[start_step], schedule[end_step]
        progress = (current_step - start_step) / float(end_step - start_step)
        
        # 阶跃参数 (Dilation) 直接取起点值
        cur_dilation = start_cfg['dilation']
        
        # 平滑参数 (Weight) 进行线性插值
        cur_dice_w = start_cfg['dice_w'] + progress * (end_cfg['dice_w'] - start_cfg['dice_w'])
        
        return cur_dilation, cur_dice_w

    def loss_by_feat(self, seg_logits: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        
        if self.training:
            self.local_step += 1
        current_step = self.local_step.item()
        
        # 【修改点 3】：不再接收 cur_thresh
        cur_dilation, cur_dice_w = self._get_dynamic_params(current_step)

        loss = dict()
        p_logit, i_logit, d_logit = seg_logits
        sem_label = self._stack_batch_gt(batch_data_samples)

        # 统一缩放尺寸
        p_logit = resize(input=p_logit, size=sem_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        i_logit = resize(input=i_logit, size=sem_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        d_logit = resize(input=d_logit, size=sem_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        
        sem_label = sem_label.squeeze(1)

        # 生成带有动态粗细的拉普拉斯边界
        bd_label = self._generate_laplacian_boundary(
            sem_label, 
            num_classes=self.num_classes, 
            ignore_index=self.ignore_index, 
            dilation_size=cur_dilation
        )

        # 1. 语义主干 Loss
        loss['loss_sem_p'] = self.loss_decode[0](p_logit, sem_label, ignore_index=self.ignore_index)
        loss['loss_sem_i'] = self.loss_decode[1](i_logit, sem_label)
        
        # =====================================================================
        # [创新点 4]: 像素级与结构级联合边界损失 (Pixel-Structure Joint Boundary Loss)
        # =====================================================================
        bce_loss = self.loss_decode[2](d_logit, bd_label)
        pred_sigmoid = torch.sigmoid(d_logit[:, 0, :, :])
        
        valid_mask = (sem_label != self.ignore_index).float()
        intersection = (pred_sigmoid * bd_label * valid_mask).sum(dim=(1, 2))
        union = (pred_sigmoid * valid_mask).sum(dim=(1, 2)) + \
                (bd_label * valid_mask).sum(dim=(1, 2))
        dice_loss = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
        
        loss['loss_bd_laplacian'] = bce_loss + cur_dice_w * dice_loss
        
        # =====================================================================
        # [创新点 5]: 神谕直连语义反哺 (Oracle-Direct Semantic Feedback)
        # =====================================================================
        filler = torch.ones_like(sem_label) * self.ignore_index
        
        # 【修改点 4】：最核心的一步！完全抛弃 pred_sigmoid > cur_thresh。
        # 直接使用绝对完美的拉普拉斯物理边界 (bd_label > 0.5) 作为监督掩码。
        # 彻底解决由于权重衰减导致的“置信度下降 -> 掩码失效 -> 梯度断裂”的死亡螺旋！
        sem_bd_label = torch.where(bd_label > 0.5, sem_label, filler)
        
        loss['loss_sem_bd'] = self.loss_decode[3](i_logit, sem_bd_label) 
        
        # 3. 准确率统计
        loss['acc_seg'] = accuracy(i_logit, sem_label, ignore_index=self.ignore_index)
            
        return loss
