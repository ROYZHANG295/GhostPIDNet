import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

raw_dir = 'data/SegNet-Tutorial/CamVid'
out_dir = 'data/camvid'

splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(f'{out_dir}/img_dir/{split}', exist_ok=True)
    os.makedirs(f'{out_dir}/ann_dir/{split}', exist_ok=True)

    img_folder = os.path.join(raw_dir, split)
    mask_folder = os.path.join(raw_dir, split + 'annot')

    if not os.path.exists(img_folder):
        continue

    filenames = os.listdir(img_folder)
    for fn in tqdm(filenames, desc=f'Processing {split}'):
        if not fn.endswith('.png'):
            continue

        # 1. 复制原图到 img_dir
        shutil.copy(os.path.join(img_folder, fn), 
                    os.path.join(out_dir, 'img_dir', split, fn))

        # 2. 转换标签：原图已经是 0-11，只需把 11 改成 255
        mask_path = os.path.join(mask_folder, fn)
        mask_np = np.array(Image.open(mask_path))
        
        # 把类别 11 (Unlabelled) 替换为 255 (mmseg的忽略常量)
        mask_np[mask_np == 11] = 255
        
        # 保存到 ann_dir
        out_path = os.path.join(out_dir, 'ann_dir', split, fn)
        Image.fromarray(mask_np).save(out_path)

print("\n✅ 重新转换完成！")
print("💡 提示：现在你去 ann_dir 里看图片，它们看起来会是黑色的（带一点点白边），这是绝对正确的 mmseg 格式！")
