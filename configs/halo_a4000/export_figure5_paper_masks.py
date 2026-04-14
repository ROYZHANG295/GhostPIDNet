import argparse
import os
import numpy as np
from PIL import Image
import torch

from mmengine.config import Config
from mmseg.apis import init_model, inference_model

# Cityscapes 官方标准的 19 类配色表 (严格对应你的 Ground Truth 颜色)
CITYSCAPES_PALETTE = [
    [128, 64, 128],  # 0: road
    [244, 35, 232],  # 1: sidewalk
    [70, 70, 70],    # 2: building
    [102, 102, 156], # 3: wall
    [190, 153, 153], # 4: fence
    [153, 153, 153], # 5: pole
    [250, 170, 30],  # 6: traffic light
    [220, 220, 0],   # 7: traffic sign
    [107, 142, 35],  # 8: vegetation
    [152, 251, 152], # 9: terrain
    [70, 130, 180],  # 10: sky
    [220, 20, 60],   # 11: person
    [255, 0, 0],     # 12: rider
    [0, 0, 142],     # 13: car
    [0, 0, 70],      # 14: truck
    [0, 60, 100],    # 15: bus
    [0, 80, 100],    # 16: train
    [0, 0, 230],     # 17: motorcycle
    [119, 11, 32]    # 18: bicycle
]

def parse_args():
    parser = argparse.ArgumentParser(description='Export pure color masks for paper visualization')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--img-dir', help='Directory containing images to infer (e.g., leftImg8bit/val/munster)')
    parser.add_argument('--out-dir', default='work_dirs/paper_masks', help='Output directory for clean masks')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 初始化模型
    print(f"Loading model from {args.checkpoint}...")
    model = init_model(args.config, args.checkpoint, device=args.device)

    # 2. 获取要测试的图片列表
    img_names = [f for f in os.listdir(args.img_dir) if f.endswith('.png') or f.endswith('.jpg')]
    
    palette = np.array(CITYSCAPES_PALETTE, dtype=np.uint8)

    print(f"Found {len(img_names)} images. Start generating pure masks...")
    
    # 3. 逐张推理并上色
    for img_name in img_names:
        img_path = os.path.join(args.img_dir, img_name)
        
        # 推理获取预测结果
        result = inference_model(model, img_path)
        
        # 提取预测的类别索引 (shape: [H, W])
        pred_sem_seg = result.pred_sem_seg.data[0].cpu().numpy()
        
        # 创建一个全黑的彩色画布 (shape: [H, W, 3])
        color_mask = np.zeros((pred_sem_seg.shape[0], pred_sem_seg.shape[1], 3), dtype=np.uint8)
        
        # 将类别索引映射为标准颜色
        for label_id in range(len(palette)):
            color_mask[pred_sem_seg == label_id] = palette[label_id]
            
        # 保存纯净的彩色图片
        out_path = os.path.join(args.out_dir, img_name)
        Image.fromarray(color_mask).save(out_path)
        print(f"Saved: {out_path}")

    print("All done! Perfect masks are ready for your paper.")

if __name__ == '__main__':
    main()


# python configs/halo_a4000/export_paper_masks.py \
#     configs/halo_3090/halo-pidnet-s-halo-same-ddr-1xb12-120k_1024x1024-cityscapes.py \
#     experiments/halo-pidnet-s-halo-same-ddr-1xb12-120k_1024x1024-cityscapes-FULL-fb_w-1_dice_w15-10-3090-79.08/best_mIoU_iter_118000.pth \
#     --img-dir data/cityscapes/leftImg8bit/val/munster \
#     --out-dir work_dirs/paper_masks_halo_pidnet_best