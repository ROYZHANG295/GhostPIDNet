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
class DDRHeadLaplacianDynamic(BaseDecodeHead):
    """
    ===========================================================================
    【论文核心故事线 (The Grand Story)】：
    
    1. 痛点 (容量悖论 Capacity Paradox)：
       DDRNet 这种实时网络参数很少。如果全程死磕高频边界，网络会耗尽算力，
       导致大块语义（如路面）识别崩溃。这就是“脑容量不够”的悖论。
       
    2. 创新 1 (零开销拉普拉斯 Zero-Cost Laplacian)：
       不用复杂的神经网络去学边缘，直接用固定的数学算子(拉普拉斯)瞬间勾勒物理边缘。
       推理时 0 负担，不掉一帧 FPS！
       
    3. 创新 2 (时空动态课程学习 Spatio-Temporal Dynamic Curriculum)：
       前期下“猛药”(高权重+粗边界)打基础；
       中期“平滑减药”(线性降权)防崩溃；
       后期“瞬间卸下沙袋”(极低权重+细边界)释放算力，冲击 SOTA！
    ===========================================================================
    """

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
            **kwargs)

        self.head = self._make_base_head(self.in_channels, self.channels)
        self.aux_head = self._make_base_head(self.in_channels // 2, self.channels)
        self.aux_cls_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)
        
        # =====================================================================
        # 【修改点 1】：新增计步器与轻量化边界预测头
        # [小白秒懂]: 
        # 1. local_step：相当于给模型戴个手表，记录跑到第几步了，方便动态调整策略。
        # 2. bd_cls_seg：一个极小的 1x1 卷积，专门用来预测单通道的黑白边界图。
        #    (注意：这个头只在训练时用，部署上线时直接扔掉，0 成本！)
        # =====================================================================
        self.register_buffer('local_step', torch.tensor(0, dtype=torch.long))
        self.bd_cls_seg = nn.Conv2d(self.channels, 1, kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs: Union[Tensor, Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        if self.training:
            c3_feat, c5_feat = inputs
            
            # Context 分支 (负责大块全局语义)
            x_c = self.head(c5_feat)
            x_c_logit = self.cls_seg(x_c)
            
            # Spatial 分支 (负责高分辨率空间细节)
            x_s = self.aux_head(c3_feat)
            x_s_logit = self.aux_cls_seg(x_s)
            
            # =================================================================
            # 【修改点 2】：训练期提取边界特征
            # [小白秒懂]: 让 Spatial 分支顺便预测一下边界 (bd_logit)
            # =================================================================
            bd_logit = self.bd_cls_seg(x_s)

            return x_c_logit, x_s_logit, bd_logit
        else:
            # 推理阶段：干干净净，没有任何多余操作，保证极速！
            x_c = self.head(inputs)
            x_c = self.cls_seg(x_c)
            return x_c

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

    # =========================================================================
    # 【修改点 3】：在线拉普拉斯边界提取
    # [小白秒懂]: 用数学公式瞬间勾勒出完美轮廓。不用存几百GB的离线边缘图，
    # 直接在显卡里用 3x3 矩阵扫一遍标签，0 延迟生成 100% 对齐的锐利边缘！
    # =========================================================================
    def _generate_laplacian_boundary(self, semantic_gt: Tensor, num_classes: int, 
                                     ignore_index: int = 255, dilation_size: int = 3) -> Tensor:
        valid_mask = (semantic_gt != ignore_index).float().unsqueeze(1)
        clean_gt = torch.where(semantic_gt == ignore_index, torch.zeros_like(semantic_gt), semantic_gt)
        
        # 把标签变成独热编码 (One-hot)
        gt_onehot = F.one_hot(clean_gt, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 魔法矩阵：拉普拉斯算子 (中间挖个坑，四周填满土，一扫就能找出边缘)
        laplacian_kernel = torch.tensor([
            [1.0,  1.0, 1.0],
            [1.0, -8.0, 1.0],
            [1.0,  1.0, 1.0]
        ], device=semantic_gt.device, dtype=torch.float32).view(1, 1, 3, 3).repeat(num_classes, 1, 1, 1)
        
        edge = F.conv2d(gt_onehot, laplacian_kernel, padding=1, groups=num_classes)
        edge = (torch.abs(edge) > 0.1).float()
        boundary_map = torch.max(edge, dim=1, keepdim=True)[0]
        
        # 动态形态学膨胀：如果 dilation_size 大，边缘就变粗（适合前期给网络降低难度）
        if dilation_size > 1:
            pad = dilation_size // 2
            boundary_map = F.max_pool2d(boundary_map, kernel_size=dilation_size, stride=1, padding=pad)
            boundary_map = boundary_map[:, :, :valid_mask.shape[2], :valid_mask.shape[3]]
            
        return (boundary_map * valid_mask).squeeze(1)

    # =========================================================================
    # 【修改点 4】：120k 时空动态课程学习调度器
    # [小白秒懂]: 打破“死记硬背”，根据训练进度自动调节“边缘粗细”和“惩罚力度”。
    # 完美适配 120k 训练总步数！
    # =========================================================================
    def _get_dynamic_params(self, current_step: int) -> Tuple[int, float, float]:
        schedule = {
            # 阶段 1 (0-40k)：强先验期。边缘画粗一点(5)，惩罚极重(3.0)，强迫打好基础。
            0:      {'step': {'dilation': 5}, 'smooth': {'dice_w': 3.0, 'thresh': 0.50}},
            40000:  {'step': {'dilation': 5}, 'smooth': {'dice_w': 3.0, 'thresh': 0.50}},
            
            # 阶段 2 (40k-80k)：平滑衰减期。边缘还是粗的，但惩罚慢慢降到 1.0，防止网络梯度休克。
            80000:  {'step': {'dilation': 5}, 'smooth': {'dice_w': 1.0, 'thresh': 0.55}},
            
            # 阶段 3 (80k-120k)：瞬间减负微调期。要求突然变高，必须画极细边缘(3)，
            # 但同时彻底松绑，惩罚降到 0.5，算力全开冲击极高 mIoU！
            80001:  {'step': {'dilation': 3}, 'smooth': {'dice_w': 0.5, 'thresh': 0.60}},
            120000: {'step': {'dilation': 3}, 'smooth': {'dice_w': 0.5, 'thresh': 0.60}} 
        }

        milestones = sorted(schedule.keys())
        if current_step <= milestones[0]:
            cfg = schedule[milestones[0]]
            return cfg['step']['dilation'], cfg['smooth']['dice_w'], cfg['smooth']['thresh']
        if current_step >= milestones[-1]:
            cfg = schedule[milestones[-1]]
            return cfg['step']['dilation'], cfg['smooth']['dice_w'], cfg['smooth']['thresh']
            
        start_step, end_step = milestones[0], milestones[-1]
        for i in range(len(milestones) - 1):
            if milestones[i] <= current_step < milestones[i+1]:
                start_step, end_step = milestones[i], milestones[i+1]
                break
                
        start_cfg, end_cfg = schedule[start_step], schedule[end_step]
        progress = (current_step - start_step) / float(end_step - start_step)
        
        cur_dilation = start_cfg['step']['dilation'] # 物理结构不平滑，直接硬切
        cur_dice_w = start_cfg['smooth']['dice_w'] + progress * (end_cfg['smooth']['dice_w'] - start_cfg['smooth']['dice_w']) # 权重必须平滑插值！
        cur_thresh = start_cfg['smooth']['thresh'] + progress * (end_cfg['smooth']['thresh'] - start_cfg['smooth']['thresh'])
        
        return cur_dilation, cur_dice_w, cur_thresh

    def loss_by_feat(self, seg_logits: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        
        if self.training:
            self.local_step += 1
        current_step = self.local_step.item()
        
        # 看看手表，获取当前的动态参数 (粗细、权重、阈值)
        cur_dilation, cur_dice_w, cur_thresh = self._get_dynamic_params(current_step)

        loss = dict()
        context_logit, spatial_logit, bd_logit = seg_logits
        seg_label = self._stack_batch_gt(batch_data_samples)

        # 统一放大到原图尺寸
        context_logit = resize(context_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        spatial_logit = resize(spatial_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        bd_logit = resize(bd_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        
        seg_label = seg_label.squeeze(1)

        # 当场生成动态粗细的完美边缘图！
        bd_label = self._generate_laplacian_boundary(
            seg_label, num_classes=self.num_classes, ignore_index=self.ignore_index, dilation_size=cur_dilation
        )

        # 计算原版 DDRNet 的 Spatial 语义损失
        loss['loss_spatial'] = self.loss_decode[1](spatial_logit, seg_label)
        
        # =====================================================================
        # 【修改点 5】：动态置信度语义反哺 + 联合边界损失
        # [小白秒懂]: 
        # 1. 反哺机制：让 Spatial 分支预测出的极高置信度边缘(>cur_thresh)，
        #    变成“硬约束”塞给 Context 分支，强迫 Context 分支在边缘处不要糊！
        # 2. 边界损失：BCE 算像素对错，Dice 算线条连贯性。配合动态权重 cur_dice_w 发挥神威！
        # =====================================================================
        pred_sigmoid = torch.sigmoid(bd_logit[:, 0, :, :])
        filler = torch.ones_like(seg_label) * self.ignore_index
        
        # 反哺 Context 分支 (宁缺毋滥，只有大于动态阈值才干预)
        context_target = torch.where(pred_sigmoid > cur_thresh, seg_label, filler)
        loss['loss_context'] = self.loss_decode[0](context_logit, context_target)

        # 计算动态边界损失
        bce_loss = F.binary_cross_entropy_with_logits(bd_logit.squeeze(1), bd_label)
        
        valid_mask = (seg_label != self.ignore_index).float()
        intersection = (pred_sigmoid * bd_label * valid_mask).sum(dim=(1, 2))
        union = (pred_sigmoid * valid_mask).sum(dim=(1, 2)) + (bd_label * valid_mask).sum(dim=(1, 2))
        dice_loss = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
        
        # 核心：将动态权重 cur_dice_w 应用于 Dice Loss
        loss['loss_bd_laplacian'] = bce_loss + cur_dice_w * dice_loss
        
        # 算个准确率汇报一下
        loss['acc_seg'] = accuracy(context_logit, seg_label, ignore_index=self.ignore_index)

        return loss
