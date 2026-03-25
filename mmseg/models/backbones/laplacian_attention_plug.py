import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS


class LaplacianChannelAttention(nn.Module):
    """LCRA: Laplacian Channel Residual Algorithm"""
    def __init__(self, channels: int, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma  # 高斯核参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 1. 计算通道间的峰值距离 (Peak Distance)
        # 对每个通道，计算其全局最大值作为特征表示
        channel_max = x.amax(dim=(2, 3))  # [B, C]
        # 计算通道间的欧氏距离矩阵
        dist = torch.cdist(channel_max.unsqueeze(2), channel_max.unsqueeze(2), p=2)  # [B, C, C]
        # 2. 高斯核构建相似性矩阵
        sim = torch.exp(-dist ** 2 / (2 * self.gamma ** 2))
        # 3. 构建拉普拉斯矩阵 L = D - A (D是度矩阵，A是邻接矩阵)
        degree = sim.sum(dim=-1, keepdim=True)  # [B, C, 1]
        laplacian = degree - sim  # [B, C, C]
        # 4. 计算通道注意力权重
        # 对原始特征进行通道维度的拉普拉斯平滑
        x_reshaped = x.view(B, C, -1)  # [B, C, H*W]
        # 这里简化为：用拉普拉斯矩阵对通道特征进行加权
        attn = torch.softmax(-laplacian, dim=-1)  # 负号确保相似通道权重高
        out = torch.bmm(attn, x_reshaped).view(B, C, H, W)
        return out + x  # 残差连接


class LaplacianSpatialAttention(nn.Module):
    """LSRA: Laplacian Spatial Residual Algorithm"""
    def __init__(self, spatial_size: int = 64, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.spatial_size = spatial_size  # 用于预计算空间位置坐标

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 1. 生成空间位置坐标 (归一化到[0,1])
        y_coords = torch.linspace(0, 1, H, device=x.device)
        x_coords = torch.linspace(0, 1, W, device=x.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)  # [H*W, 2]
        # 2. 计算空间位置间的距离矩阵
        dist = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0), p=2)  # [1, H*W, H*W]
        dist = dist.repeat(B, 1, 1)  # [B, H*W, H*W]
        # 3. 高斯核构建相似性矩阵
        sim = torch.exp(-dist ** 2 / (2 * self.gamma ** 2))
        # 4. 构建拉普拉斯矩阵
        degree = sim.sum(dim=-1, keepdim=True)
        laplacian = degree - sim
        # 5. 计算空间注意力权重
        x_reshaped = x.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        attn = torch.softmax(-laplacian, dim=-1)
        out = torch.bmm(attn, x_reshaped).permute(0, 2, 1).view(B, C, H, W)
        return out + x  # 残差连接


@MODELS.register_module()
class LaplacianAttention(nn.Module):
    """Laplacian Attention: 融合LCRA和LSRA"""
    def __init__(self, channels: int, gamma_c: float = 1.0, gamma_s: float = 1.0):
        super().__init__()
        self.channel_attn = LaplacianChannelAttention(channels, gamma_c)
        self.spatial_attn = LaplacianSpatialAttention(gamma=gamma_s)
        # 自适应融合权重 (无参数，通过动态计算)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=False)  # 初始化为0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 分别计算通道和空间注意力
        f_c = self.channel_attn(x)
        f_s = self.spatial_attn(x)
        # 自适应融合：这里简化为动态计算alpha (根据特征的方差)
        # 论文中提到"adaptive-scale allocation"，这里用通道和空间特征的方差比作为权重
        var_c = f_c.var(dim=(2, 3), keepdim=True).mean()
        var_s = f_s.var(dim=(2, 3), keepdim=True).mean()
        alpha = var_c / (var_c + var_s + 1e-8)
        out = alpha * f_c + (1 - alpha) * f_s
        return out