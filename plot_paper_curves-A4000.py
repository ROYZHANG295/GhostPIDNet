import json
import os
import matplotlib.pyplot as plt

# ================= 1. 核心配置区 (在这里修改标签和路径) =================
# 格式: {"你想在图表上显示的标签名字": "对应的 scalars.json 路径"}
# 支持任意数量，直接在下面加行即可！
LOG_CONFIG = {
    # "Baseline (PIDNet)": "work_dirs/test-0319-4-20k-pidnet-s-no-pretrained_baseline_warmup-2xb6-120k_1024x1024-cityscapes/20260319_034706/vis_data/scalars.json",
    # "Laplacian Loss": "work_dirs/test-0321-1-20k-pidnet-s-no-pretrained_laplacian_loss_warmup-2xb6-120k_1024x1024-cityscapes/20260320_042324/vis_data/scalars.json",
    # "Ours (Dual-Laplacian)": "work_dirs/test-0319-5-20k-pidnet-s-no-pretrained-dual-laplacian-attention-I-D-branch-warmup-2xb6-120k_1024x1024-cityscapes/20260319_084621/vis_data/scalars.json",
    # "Baseline": "work_dirs/test-0320-6-pidnet-s-no-pretrained_baseline_warmup-2xb6-120k_1024x1024-cityscapes/20260320_095811/vis_data/scalars.json",
    #"Baseline": "work_dirs/test-0320-6-pidnet-s-no-pretrained_baseline_warmup-2xb6-120k_1024x1024-cityscapes/20260320_164727-76.39/vis_data/scalars.json",
    # "Lap Plug": "work_dirs/test-0325-1-pidnet-s-no-pretrained_laplacian_attention_plug_warmup-2xb6-120k_1024x1024-cityscapes/20260325_175033/vis_data/scalars.json",
    #"Lap Loss Dynamic3-seed42": "work_dirs/test-0325-3-120k-pidnet-s-no-pretrained_laplacian_loss_opt3_dynamic3_warmup-2xb6-120k_1024x1024-cityscape-seed42/20260325_190800/vis_data/scalars.json",
   #"Laplacian Loss Dynamic":"work_dirs/test-0322-1-120k-pidnet-s-no-pretrained_laplacian_loss_opt3_dynamic_warmup-2xb6-120k_1024x1024-cityscapes/20260321_223222/vis_data/scalars.json",
    #"Laplacian Loss":"work_dirs/test-0321-3-120k-pidnet-s-no-pretrained_laplacian_loss_opt3_warmup-2xb6-120k_1024x1024-cityscapes/20260321_013912/vis_data/scalars.json"
    #"Laplacian Loss Dynamic 1": "work_dirs/test-0322-1-120k-pidnet-s-no-pretrained_laplacian_loss_opt3_dynamic_warmup-2xb6-120k_1024x1024-cityscapes/20260322_051110/vis_data/scalars.json",
    #"Laplacian Loss Dynamic 2": "work_dirs/test-0323-1-120k-pidnet-s-no-pretrained_laplacian_loss_opt3_dynamic2_warmup-2xb6-120k_1024x1024-cityscapes/20260322_231131/vis_data/scalars.json"
    # "Laplacian Loss Dynamic 3": "work_dirs/test-0323-2-120k-pidnet-s-no-pretrained_laplacian_loss_opt3_dynamic3_warmup-2xb6-120k_1024x1024-cityscapes/20260323_080513/vis_data/scalars.json",
    #"Baseline with 78.36": "../mmsegmentation/work_dirs/pidnet-s_2xb6-120k_1024x1024-cityscapes-runable-weight-class/20260214_210327-12k-miou=78.34/vis_data/scalars.json",
    #"Baseline CE weight": "work_dirs/test-0324-2-pidnet-s-cityspace-baseline-120k-1GPU-crossentropy-weight-b6/20260323_225044/vis_data/scalars.json",
    #"Baseline weight": "work_dirs/test-0324-4-pidnet-s_2xb6-120k_1024x1024-cityscapes-runable-weight-class/20260324_114111/vis_data/scalars.json",
    #"Distance Lap": "work_dirs/test-0318-2-pidnet-s-no-pretrained-distance-edge-laplacian-attention-I-branch-zero-warmup_2xb6-120k_1024x1024-cityscapes-runable-weight-class-76.62/20260317_114644/vis_data/scalars.json",
    #"Lap I":"work_dirs/test-0318-3-pidnet-s-no-pretrained-laplacian-attention-I-branch-zero-norm-warmup-2xb6-120k_1024x1024-cityscapes-runable-weight-class-77.26/20260318_042453/vis_data/scalars.json"
    # 如果你后续跑了消融实验，只需要取消下面注释并填上路径：
    # "Ours (Only I-Branch)": "work_dirs/你的I分支实验路径/vis_data/scalars.json",
    # "Ours (Only D-Branch)": "work_dirs/你的D分支实验路径/vis_data/scalars.json",
    # "baseline seed304": "work_dirs/test-0320-6-pidnet-s-no-pretrained_baseline_warmup-2xb6-120k_1024x1024-cityscapes/20260320_164727-76.39/vis_data/scalars.json",
    # "baseline seed42": "work_dirs/test-0325-3-2-pidnet-s-no-pretrained_baseline_warmup-2xb6-120k_1024x1024-cityscapes-seed42/20260325_235148/vis_data/scalars.json",
    # "Lap dynamic3 seed42": "work_dirs/test-0325-3-1-120k-pidnet-s-no-pretrained_laplacian_loss_opt3_dynamic3_warmup-2xb6-120k_1024x1024-cityscape-seed42/20260325_235117/vis_data/scalars.json",
    
    
    # "20k_baseline": "work_dirs/test-0319-4-20k-pidnet-s-no-pretrained_baseline_warmup-2xb6-120k_1024x1024-cityscapes/20260319_034706/vis_data/scalars.json",
    # # "20k_lap_plug": "work_dirs/test-0325-3-20k-pidnet-s-no-pretrained_laplacian_attention_plug_warmup-2xb6-120k_1024x1024-cityscapes/20260326_100258/vis_data/scalars.json",
    # "20k_edge_aware": "work_dirs/test-0327-2-pidnet-s-no-pretrained_edge_aware_warmup-2xb6-120k_1024x1024-cityscapes-20k/20260327_073314/vis_data/scalars.json",

    "baseline b12": "experiments/test-0324-1-120k-b12-1GPU-pidnet-s-cityspace-baseline/20260324_000733/vis_data/scalars.json",
    #"Lap loss v1": "experiments/test-0322-2-120k-pidnet-s-with-pretrained_laplacian_loss_opt3_warmup-2xb6-120k_1024x1024-cityscapes-3090-78.65/20260322_231459-78.65/vis_data/scalars.json",
    "Lap Dynamic3 b12": "work_dirs/4-laplacian-loss-dynamic3-1GPU-120k-pidnet-s_1xb12-120k_1024x1024-cityscapes/20260328_070424/vis_data/scalars.json",

# "Baseline 76.39": "work_dirs/baselines/seed304-76.39/scalars.json",
# "Lap Plug": "work_dirs/3-pidnet-s-no-pretrained_laplacian_attention_plug_warmup-2xb6-120k_1024x1024-cityscapes/20260326_121148/vis_data/scalars.json",
}

# 论文常用高级配色 (蓝, 红, 绿, 橙, 紫, 棕) 和 不同的点标记
COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
MARKERS = ['o', 'o','^', 's',  'D', 'v', 'p']
# ========================================================================

def load_log_data(file_path):
    """读取 MMSegmentation 的 json 日志，分离 loss 和 mIoU"""
    steps_loss, losses = [], []
    steps_miou, mious = [], []
    
    if not os.path.exists(file_path):
        print(f"⚠️ 警告: 找不到文件 -> {file_path}")
        return [], [], [], []
        
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                step = data.get('step', 0)
                
                if 'loss' in data:
                    steps_loss.append(step)
                    losses.append(data['loss'])
                if 'mIoU' in data:
                    steps_miou.append(step)
                    mious.append(data['mIoU'])
            except:
                continue
    return steps_loss, losses, steps_miou, mious

def smooth_curve(scalars, weight=0.85):
    """曲线平滑算法"""
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# ================= 2. 开始画图 =================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# 遍历配置字典，动态画图
for i, (label_name, json_path) in enumerate(LOG_CONFIG.items()):
    # 动态获取颜色和标记（循环使用，防止越界）
    color = COLORS[i % len(COLORS)]
    marker = MARKERS[i % len(MARKERS)]
    
    # 读取数据
    step_loss, loss, step_miou, miou = load_log_data(json_path)
    
    if not loss and not miou:
        continue # 如果文件空或不存在，跳过
        
    # ------ 左图：画 Loss 曲线 ------
    if loss:
        # 画半透明原始毛刺线
        ax1.plot(step_loss, loss, color=color, alpha=0.15)
        # 画加粗平滑线
        ax1.plot(step_loss, smooth_curve(loss, 0.9), label=label_name, color=color, linewidth=2)

    # ------ 右图：画 mIoU 曲线 ------
    if miou:
        ax2.plot(step_miou, miou, label=label_name, color=color, linewidth=1, marker=marker, markersize=3)

# ================= 3. 图表排版与美化 =================
# Loss 图设置
ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('Iterations', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=11)

# mIoU 图设置
ax2.set_title('Validation mIoU Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Iterations', fontsize=12)
ax2.set_ylabel('mIoU (%)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=11, loc='lower right')

plt.tight_layout()

# 保存为高清图片
save_png = 'multi_curves_comparison_A4000.png'
save_pdf = 'multi_curves_comparison_A4000.pdf'
plt.savefig(save_png, dpi=300, bbox_inches='tight')
plt.savefig(save_pdf, bbox_inches='tight')

print(f"\n✅ 画图成功！共对比了 {len(LOG_CONFIG)} 组实验。")
print(f"图片已保存为: {save_png} 和 {save_pdf}")
