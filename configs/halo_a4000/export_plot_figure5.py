import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os

# =====================================================================
# === 1. 用户配置区 ===
# =====================================================================

# 你的图片存放目录
BASE_DIR = "work_dirs/vis_results" 

# 配置每一行的样本 (这里以 2 行为例，你可以加到 3-4 行)
# zoom_box: (x_min, y_min, x_max, y_max) 是你想局部放大的区域坐标
SAMPLES = [
    {
        'img': 'munster_000000_000019_leftImg8bit.png',
        'gt':  'munster_000000_000019_gtFine_color.png',
        'pid_base': 'pid_base/munster_000000_000019.png',
        'pid_halo': 'pid_halo/munster_000000_000019.png',
        'ddr_base': 'ddr_base/munster_000000_000019.png',
        'ddr_halo': 'ddr_halo/munster_000000_000019.png',
        'zoom_box': (800, 300, 1000, 500)  # 示例：框出右上角的红绿灯或电线杆
    },
    {
        'img': 'frankfurt_000001_013016_leftImg8bit.png',
        'gt':  'frankfurt_000001_013016_gtFine_color.png',
        'pid_base': 'pid_base/frankfurt_000001_013016.png',
        'pid_halo': 'pid_halo/frankfurt_000001_013016.png',
        'ddr_base': 'ddr_base/frankfurt_000001_013016.png',
        'ddr_halo': 'ddr_halo/frankfurt_000001_013016.png',
        'zoom_box': (400, 400, 600, 600)  # 示例：框出中间的行人/自行车
    }
]

TITLES = ["Image", "Ground Truth", "PIDNet Baseline", "HALO-PIDNet", "DDRNet Baseline", "HALO-DDRNet"]

# =====================================================================
# === 2. 图像处理核心函数 ===
# =====================================================================

def add_zoom_inset(img_path, box, zoom_ratio=2.5, inset_pos='bottom_right'):
    """读取图片，画红框，并在右下角添加放大的画中画"""
    # 1. 读取原图
    if not os.path.exists(img_path):
        # 如果文件不存在，生成一张纯黑图占位（防止报错）
        print(f"Warning: Not found {img_path}")
        return Image.new('RGB', (1024, 512), (50, 50, 50))
    
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    x1, y1, x2, y2 = box
    
    # 2. 截取局部并放大
    crop_img = img.crop(box)
    new_w, new_h = int((x2 - x1) * zoom_ratio), int((y2 - y1) * zoom_ratio)
    crop_img = crop_img.resize((new_w, new_h), Image.NEAREST) # NEAREST 保持分割边缘锐利
    
    # 3. 给放大的画中画加一个粗红边框
    crop_with_border = Image.new('RGB', (new_w + 6, new_h + 6), (255, 0, 0))
    crop_with_border.paste(crop_img, (3, 3))
    
    # 4. 把画中画贴到原图右下角
    img_w, img_h = img.size
    paste_x = img_w - new_w - 6 - 10 # 距离右边缘 10 像素
    paste_y = img_h - new_h - 6 - 10 # 距离下边缘 10 像素
    img.paste(crop_with_border, (paste_x, paste_y))
    
    # 5. 在原图上画出红色的虚线框/实线框指示放大区域
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
    
    # 画一条连接线 (可选)
    draw.line([(x2, y2), (paste_x, paste_y)], fill=(255, 0, 0), width=2)
    
    return img

# =====================================================================
# === 3. 拼图与渲染 ===
# =====================================================================

# 假设所有图大小一致，比如 Cityscapes 是 2048x1024，我们缩小一半排版
TARGET_W, TARGET_H = 1024, 512  

# 创建大画布 (宽 = 6列 * 单宽，高 = 行数 * 单高 + 标题高度)
margin_top = 60
margin_between = 10
total_w = 6 * TARGET_W + 5 * margin_between
total_h = len(SAMPLES) * TARGET_H + (len(SAMPLES) - 1) * margin_between + margin_top

canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

# 如果你的系统有字体，可以加载；没有的话用默认
try:
    font = ImageFont.truetype("arial.ttf", 40)
except:
    font = ImageFont.load_default()

# 1. 画标题
for col, title in enumerate(TITLES):
    x = col * (TARGET_W + margin_between) + TARGET_W // 2
    # 居中写字 (简单近似居中)
    draw.text((x - len(title)*10, 10), title, fill=(0, 0, 0), font=font)

# 2. 贴图片
keys = ['img', 'gt', 'pid_base', 'pid_halo', 'ddr_base', 'ddr_halo']

for row, sample in enumerate(SAMPLES):
    box = sample['zoom_box']
    for col, key in enumerate(keys):
        img_path = os.path.join(BASE_DIR, sample[key])
        
        # 处理图片（加红框和画中画）
        processed_img = add_zoom_inset(img_path, box, zoom_ratio=2.0)
        processed_img = processed_img.resize((TARGET_W, TARGET_H), Image.ANTIALIAS)
        
        # 计算粘贴坐标
        paste_x = col * (TARGET_W + margin_between)
        paste_y = margin_top + row * (TARGET_H + margin_between)
        
        canvas.paste(processed_img, (paste_x, paste_y))

# 3. 保存高清图
canvas.save("figure5_qualitative_comparison.jpg", quality=95)
canvas.save("figure5_qualitative_comparison.pdf", resolution=300)
print("Figure 5 保存成功！")
