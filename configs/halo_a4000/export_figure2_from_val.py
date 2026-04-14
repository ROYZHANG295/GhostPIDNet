import argparse
import os
import os.path as osp
from typing import Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmseg.registry import DATASETS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export Figure 2 (OLB + Dynamic Dilation) from a val sample')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument(
        '--sample-idx',
        type=int,
        default=0,
        help='Index of the validation sample')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='work_dirs/figure2_single',
        help='Output directory')
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='cpu or cuda:0')
    parser.add_argument(
        '--dilations',
        type=int,
        nargs='+',
        default=[5, 4, 3],
        help='Dilations to visualize, e.g. --dilations 5 4 3')
    parser.add_argument(
        '--save-input',
        action='store_true',
        help='Also save the raw input image if img_path exists')
    return parser.parse_args()


def get_decode_head_cfg(cfg):
    decode_head_cfg = cfg.model.decode_head
    if isinstance(decode_head_cfg, (list, tuple)):
        decode_head_cfg = decode_head_cfg[0]
    return decode_head_cfg


def apply_label_map_and_reduce_zero_label(
        gt: np.ndarray,
        label_map: Optional[dict] = None,
        reduce_zero_label: bool = False) -> np.ndarray:
    """Mimic mmseg annotation processing logic when needed."""
    gt = gt.copy()

    if label_map is not None and len(label_map) > 0:
        gt_copy = gt.copy()
        for old_id, new_id in label_map.items():
            gt[gt_copy == old_id] = new_id

    if reduce_zero_label:
        gt = gt.astype(np.int32)
        gt[gt == 0] = 255
        gt = gt - 1
        gt[gt == 254] = 255
        gt = gt.astype(np.uint8)

    return gt


def load_raw_sample_from_dataset(dataset, sample_idx: int):
    """Load raw img path and seg map path from dataset data_info."""
    if hasattr(dataset, 'get_data_info'):
        info = dataset.get_data_info(sample_idx)
    else:
        raise RuntimeError('Dataset does not support get_data_info().')

    img_path = info.get('img_path', None)
    seg_map_path = info.get('seg_map_path', None)

    # Some datasets may use another field name
    if seg_map_path is None:
        seg_map_path = info.get('ann_path', None)

    if seg_map_path is None:
        raise RuntimeError('Cannot find seg_map_path/ann_path in dataset data_info.')

    return img_path, seg_map_path


def load_gt_label(dataset, sample_idx: int) -> Tuple[np.ndarray, Optional[str], str]:
    """Load raw GT label map from dataset."""
    img_path, seg_map_path = load_raw_sample_from_dataset(dataset, sample_idx)

    gt = np.array(Image.open(seg_map_path), dtype=np.uint8)

    label_map = getattr(dataset, 'label_map', None)
    reduce_zero_label = getattr(dataset, 'reduce_zero_label', False)

    gt = apply_label_map_and_reduce_zero_label(
        gt,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label
    )

    return gt, img_path, seg_map_path


def extract_raw_laplacian_boundary(
        semantic_gt: torch.Tensor,
        num_classes: int,
        ignore_index: int = 255) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    semantic_gt: [B, H, W]
    returns:
        raw_boundary: [B, H, W]
        valid_mask: [B, H, W]
    """
    valid_mask = (semantic_gt != ignore_index).float()

    clean_gt = torch.where(
        semantic_gt == ignore_index,
        torch.zeros_like(semantic_gt),
        semantic_gt
    )

    gt_onehot = F.one_hot(clean_gt.long(), num_classes=num_classes)
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

    boundary_map = boundary_map * valid_mask.unsqueeze(1)

    return boundary_map.squeeze(1), valid_mask


def apply_boundary_dilation(
        boundary_map: torch.Tensor,
        valid_mask: torch.Tensor,
        dilation_size: int) -> torch.Tensor:
    """
    boundary_map: [B, H, W]
    valid_mask:   [B, H, W]
    """
    boundary_map = boundary_map.unsqueeze(1)
    valid_mask = valid_mask.unsqueeze(1)

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


def label_to_color(
        label: np.ndarray,
        palette: Optional[list],
        ignore_index: int = 255,
        ignore_color=(0, 0, 0)) -> np.ndarray:
    """
    Convert label map [H, W] to color image [H, W, 3].
    """
    h, w = label.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    if palette is None or len(palette) == 0:
        # fallback: matplotlib tab20
        cmap = plt.get_cmap('tab20')
        unique_ids = np.unique(label)
        for cls_id in unique_ids:
            if cls_id == ignore_index:
                continue
            rgb = np.array(cmap(int(cls_id) % 20)[:3]) * 255
            color[label == cls_id] = rgb.astype(np.uint8)
    else:
        for cls_id, rgb in enumerate(palette):
            color[label == cls_id] = np.array(rgb, dtype=np.uint8)

    color[label == ignore_index] = np.array(ignore_color, dtype=np.uint8)
    return color


def save_gray(path: str, arr: np.ndarray):
    plt.imsave(path, arr, cmap='gray', vmin=0, vmax=1)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmseg'))

    # Build val dataset
    dataset = DATASETS.build(cfg.val_dataloader.dataset)

    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        raise ValueError(
            f'sample_idx {args.sample_idx} out of range, dataset size = {len(dataset)}')

    # Load GT and paths
    gt_np, img_path, seg_map_path = load_gt_label(dataset, args.sample_idx)

    decode_head_cfg = get_decode_head_cfg(cfg)
    num_classes = decode_head_cfg.get(
        'num_classes',
        len(dataset.metainfo.get('classes', []))
    )
    ignore_index = decode_head_cfg.get('ignore_index', 255)

    device = torch.device(args.device)

    sem_label = torch.from_numpy(gt_np).long().unsqueeze(0).to(device)  # [1, H, W]

    with torch.no_grad():
        raw_boundary, valid_mask = extract_raw_laplacian_boundary(
            sem_label,
            num_classes=num_classes,
            ignore_index=ignore_index
        )

        dilated_maps = {}
        for d in args.dilations:
            dilated_maps[d] = apply_boundary_dilation(raw_boundary, valid_mask, dilation_size=d)

    # Convert to numpy
    sem_np = sem_label[0].detach().cpu().numpy().copy()
    valid_np = valid_mask[0].detach().cpu().numpy()
    raw_np = raw_boundary[0].detach().cpu().numpy()

    # Save semantic label with palette
    palette = dataset.metainfo.get('palette', None)
    sem_color = label_to_color(sem_np, palette=palette, ignore_index=ignore_index)

    Image.fromarray(sem_color).save(osp.join(args.out_dir, 'semantic_label.png'))
    save_gray(osp.join(args.out_dir, 'valid_mask.png'), valid_np)
    save_gray(osp.join(args.out_dir, 'olb_boundary.png'), raw_np)

    # Save each dilation
    dilated_np = {}
    for d in args.dilations:
        d_np = dilated_maps[d][0].detach().cpu().numpy()
        dilated_np[d] = d_np
        save_gray(osp.join(args.out_dir, f'dilation_{d}.png'), d_np)

    # Save raw input image if available
    input_img = None
    if args.save_input and img_path is not None and osp.exists(img_path):
        input_img = Image.open(img_path).convert('RGB')
        input_img.save(osp.join(args.out_dir, 'input_image.png'))

    # Save combined panel
    # Default panel order:
    # Semantic Label | Valid Mask | OLB Boundary | d=5 | d=4 | d=3
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))

    axes[0].imshow(sem_color)
    axes[0].set_title('Semantic Label')

    axes[1].imshow(valid_np, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    axes[1].set_title('Valid Mask')

    axes[2].imshow(raw_np, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    axes[2].set_title('OLB Boundary')

    # If user passed other dilations, still try to map titles in order
    for i, d in enumerate(args.dilations[:3]):
        axes[3 + i].imshow(dilated_np[d], cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        axes[3 + i].set_title(f'Stage {i + 1}: d={d}')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(
        osp.join(args.out_dir, 'figure2_panel.png'),
        dpi=200,
        bbox_inches='tight'
    )
    plt.close(fig)

    # Save a simple meta text
    with open(osp.join(args.out_dir, 'meta.txt'), 'w', encoding='utf-8') as f:
        f.write(f'config: {args.config}\n')
        f.write(f'sample_idx: {args.sample_idx}\n')
        f.write(f'img_path: {img_path}\n')
        f.write(f'seg_map_path: {seg_map_path}\n')
        f.write(f'num_classes: {num_classes}\n')
        f.write(f'ignore_index: {ignore_index}\n')
        f.write(f'dilations: {args.dilations}\n')

    print('Done. Files saved to:', args.out_dir)


if __name__ == '__main__':
    main()

# python configs/halo_a4000/export_figure2_from_val.py configs/halo_a4000/halo-pidnet-s-halo-same-ddr-1xb12-120k_1024x1024-cityscapes-best-run2.py --sample-idx 0 --out-dir work_dirs/figure2_single