# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F  # 【修改点1】引入 functional 模块
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList


class BasePIDHead(BaseModule):
    """Base class for PID head.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of output channels.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
        init_cfg (dict or list[dict], optional): Init config dict.
            Default: None.
    """

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
            order=('norm', 'act', 'conv'))
        _, self.norm = build_norm_layer(norm_cfg, num_features=channels)
        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor, cls_seg: Optional[nn.Module]) -> Tensor:
        """Forward function.
        Args:
            x (Tensor): Input tensor.
            cls_seg (nn.Module, optional): The classification head.

        Returns:
            Tensor: Output tensor.
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if cls_seg is not None:
            x = cls_seg(x)
        return x


@MODELS.register_module()
class PIDHeadLaplacian(BaseDecodeHead):
    """Decode head for PIDNet.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of output channels.
        num_classes (int): Number of classes.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
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
        self.i_head = BasePIDHead(in_channels, channels, norm_cfg, act_cfg)
        self.p_head = BasePIDHead(in_channels // 2, channels, norm_cfg,
                                  act_cfg)
        self.d_head = BasePIDHead(
            in_channels // 2,
            in_channels // 4,
            norm_cfg,
        )
        self.p_cls_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        self.d_cls_seg = nn.Conv2d(in_channels // 4, 1, kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
            self,
            inputs: Union[Tensor,
                          Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.
        Args:
            inputs (Tensor | tuple[Tensor]): Input tensor or tuple of
                Tensor. When training, the input is a tuple of three tensors,
                (p_feat, i_feat, d_feat), and the output is a tuple of three
                tensors, (p_seg_logit, i_seg_logit, d_seg_logit).
                When inference, only the head of integral branch is used, and
                input is a tensor of integral feature map, and the output is
                the segmentation logit.

        Returns:
            Tensor | tuple[Tensor]: Output tensor or tuple of tensors.
        """
        if self.training:
            x_p, x_i, x_d = inputs
            x_p = self.p_head(x_p, self.p_cls_seg)
            x_i = self.i_head(x_i, self.cls_seg)
            x_d = self.d_head(x_d, self.d_cls_seg)
            return x_p, x_i, x_d
        else:
            return self.i_head(inputs, self.cls_seg)

    # def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tuple[Tensor]:
    #     gt_semantic_segs = [
    #         data_sample.gt_sem_seg.data for data_sample in batch_data_samples
    #     ]
    #     gt_edge_segs = [
    #         data_sample.gt_edge_map.data for data_sample in batch_data_samples
    #     ]
    #     gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
    #     gt_edge_segs = torch.stack(gt_edge_segs, dim=0)
    #     return gt_sem_segs, gt_edge_segs
    # 【修改点2】重写批次真值堆叠函数，移除对 gt_edge_map 的依赖
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
        return gt_sem_segs

    # 【修改点3】新增拉普拉斯边界生成函数
    def _generate_laplacian_boundary(self, semantic_gt, num_classes, ignore_index=255, dilation_size=3):
        """动态生成拉普拉斯边界"""
        # 1. 处理 ignore_index (防止 one_hot 越界报错)
        valid_mask = (semantic_gt != ignore_index).float().unsqueeze(1)
        clean_gt = torch.where(semantic_gt == ignore_index, torch.zeros_like(semantic_gt), semantic_gt)
        
        # 2. 转为 One-hot 编码 [B, C, H, W]
        gt_onehot = F.one_hot(clean_gt, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 3. 定义并应用拉普拉斯卷积核
        laplacian_kernel = torch.tensor([
            [1.0,  1.0, 1.0],
            [1.0, -8.0, 1.0],
            [1.0,  1.0, 1.0]
        ], device=semantic_gt.device, dtype=torch.float32).view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(num_classes, 1, 1, 1)
        
        # 使用分组卷积独立提取每个类别的边缘
        edge = F.conv2d(gt_onehot, laplacian_kernel, padding=1, groups=num_classes)
        edge = (torch.abs(edge) > 0.1).float()
        
        # 4. 合并所有类别的边缘
        boundary_map = torch.max(edge, dim=1, keepdim=True)[0]
        
        # 5. 形态学膨胀 (解决单像素边缘难以优化的问题)
        if dilation_size > 1:
            pad = dilation_size // 2
            boundary_map = F.max_pool2d(boundary_map, kernel_size=dilation_size, stride=1, padding=pad)
            
        # 6. 屏蔽掉 ignore 区域的边界
        boundary_map = boundary_map * valid_mask
        
        # 返回 [B, H, W] 以匹配原版 bd_label 的形状
        return boundary_map.squeeze(1)

    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        loss = dict()
        p_logit, i_logit, d_logit = seg_logits

        # sem_label, bd_label = self._stack_batch_gt(batch_data_samples)
        # 只获取语义标签，不再获取 bd_label
        sem_label = self._stack_batch_gt(batch_data_samples)

        p_logit = resize(
            input=p_logit,
            size=sem_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        i_logit = resize(
            input=i_logit,
            size=sem_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        d_logit = resize(
            input=d_logit,
            # size=bd_label.shape[2:],
            size=sem_label.shape[2:], # 尺寸对齐到语义标签
            mode='bilinear',
            align_corners=self.align_corners)
        sem_label = sem_label.squeeze(1)
        # bd_label = bd_label.squeeze(1)

        # ---------------------------------------------------------
        # 核心修改：在这里动态生成拉普拉斯边界真值
        # dilation_size 默认设为 3，若图像分辨率极大可尝试设为 5
        # ---------------------------------------------------------
        bd_label = self._generate_laplacian_boundary(
            sem_label, 
            num_classes=self.num_classes, 
            ignore_index=self.ignore_index, 
            dilation_size=3
        )

        loss['loss_sem_p'] = self.loss_decode[0](
            p_logit, sem_label, ignore_index=self.ignore_index)
        loss['loss_sem_i'] = self.loss_decode[1](i_logit, sem_label)
        loss['loss_bd_laplacian'] = self.loss_decode[2](d_logit, bd_label)
        filler = torch.ones_like(sem_label) * self.ignore_index
        sem_bd_label = torch.where(
            torch.sigmoid(d_logit[:, 0, :, :]) > 0.8, sem_label, filler)
        loss['loss_sem_bd'] = self.loss_decode[3](i_logit, sem_bd_label)
        loss['acc_seg'] = accuracy(
            i_logit, sem_label, ignore_index=self.ignore_index)
        return loss
