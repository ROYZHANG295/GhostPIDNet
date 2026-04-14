import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# === 1. 用户配置区 (在这里填入你最终实验的真实数值) ===
# =====================================================================

# 迭代总数与分界点
MAX_ITERS = 120000
T1 = 40000  # 第一阶段结束点
T2 = 80000  # 第二阶段结束点
iters = [0, T1, T2, MAX_ITERS]

# --- PIDNet-S 调度参数 (Heavy Architecture) ---
# 分别对应: [Stage 1, Stage 2, Stage 3]
pidnet_dice_w = [3.0, 1.5, 1.0]  # Boundary Dice Loss 权重
pidnet_fb_w   = [1.0, 1.0, 1.0]  # Feedback Loss 权重
pidnet_dil    = [5, 4, 3]        # 动态膨胀 Dilation size

# --- DDRNet-23-slim 调度参数 (Lightweight Architecture) ---
# 分别对应: [Stage 1, Stage 2, Stage 3]
ddrnet_dice_w = [1.0, 0.5, 0.1]  # 示例数值，请填入你 DDRNet 真实用的衰减值
ddrnet_fb_w   = [1.0, 0.5, 0.1]  # 示例数值，请填入你 DDRNet 真实用的衰减值
ddrnet_dil    = [5, 4, 3]

# =====================================================================
# === 2. 绘图辅助函数 ===
# =====================================================================

def make_step_data(x_milestones, y_values):
    """将分段数值转换为绘制阶梯图(Step plot)的坐标点"""
    x_coords = []
    y_coords = []
    for i in range(len(y_values)):
        x_coords.extend([x_milestones[i], x_milestones[i+1]])
        y_coords.extend([y_values[i], y_values[i]])
    return x_coords, y_coords

# 全局字体和清晰度设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# 论文常用配色 (Nature/Science 风格色系)
COLOR_DICE = '#E64B35'  # 红色系
COLOR_FB   = '#4DBBD5'  # 蓝色系
COLOR_DIL  = '#00A087'  # 蓝绿色系

# =====================================================================
# === 3. 开始绘图 ===
# =====================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# ---------------------------------------------------------------------
# 子图 1: PIDNet-S
# ---------------------------------------------------------------------
# 获取阶梯坐标
x_pid, y_pid_dice = make_step_data(iters, pidnet_dice_w)
_, y_pid_fb = make_step_data(iters, pidnet_fb_w)
_, y_pid_dil = make_step_data(iters, pidnet_dil)

# 绘制左轴 (Loss Weights)
line1, = ax1.plot(x_pid, y_pid_dice, color=COLOR_DICE, linewidth=2.5, linestyle='-', label='Boundary Weight ($w_{dice}$)')
line2, = ax1.plot(x_pid, y_pid_fb, color=COLOR_FB, linewidth=2.5, linestyle='--', label='Feedback Weight ($w_{fb}$)')

ax1.set_title('(a) PIDNet-S (Heavy)', fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel('Training Iterations', fontsize=12)
ax1.set_ylabel('Loss Weight', fontsize=12)
ax1.set_xlim(0, MAX_ITERS)
ax1.set_ylim(-0.2, 3.5) # 统一左轴范围
ax1.set_xticks([0, 40000, 80000, 120000])
ax1.set_xticklabels(['0', '40k', '80k', '120k'])
ax1.grid(True, linestyle=':', alpha=0.6)

# 绘制右轴 (Dilation Size)
ax1_twin = ax1.twinx()
line3, = ax1_twin.plot(x_pid, y_pid_dil, color=COLOR_DIL, linewidth=2.5, linestyle='-.', label='Dilation Size ($d$)')
ax1_twin.set_ylabel('Dilation Size ($d$)', fontsize=12)
ax1_twin.set_ylim(2, 6)
ax1_twin.set_yticks([3, 4, 5])

# 添加文字注释 (突出 PIDNet 的特点)
# ax1.text(60000, 2.5, 'Sustained Structural\nGuidance', fontsize=12, 
#          ha='right', va='center', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.9))
ax1.text(94000, 1.8, 'Sustained Structural\nGuidance', fontsize=12, color='#333333',
         ha='center', va='center', fontweight='bold')
# ---------------------------------------------------------------------
# 子图 2: DDRNet-23-slim
# ---------------------------------------------------------------------
# 获取阶梯坐标
x_ddr, y_ddr_dice = make_step_data(iters, ddrnet_dice_w)
_, y_ddr_fb = make_step_data(iters, ddrnet_fb_w)
_, y_ddr_dil = make_step_data(iters, ddrnet_dil)

# 绘制左轴 (Loss Weights)
ax2.plot(x_ddr, y_ddr_dice, color=COLOR_DICE, linewidth=2.5, linestyle='-')
ax2.plot(x_ddr, y_ddr_fb, color=COLOR_FB, linewidth=2.5, linestyle='--')

ax2.set_title('(b) DDRNet-23-slim (Lightweight)', fontsize=14, fontweight='bold', pad=10)
ax2.set_xlabel('Training Iterations', fontsize=12)
ax2.set_ylabel('Loss Weight', fontsize=12)
ax2.set_xlim(0, MAX_ITERS)
ax2.set_ylim(-0.2, 3.5) # 统一左轴范围，方便对比
ax2.set_xticks([0, 40000, 80000, 120000])
ax2.set_xticklabels(['0', '40k', '80k', '120k'])
ax2.grid(True, linestyle=':', alpha=0.6)

# 绘制右轴 (Dilation Size)
ax2_twin = ax2.twinx()
ax2_twin.plot(x_ddr, y_ddr_dil, color=COLOR_DIL, linewidth=2.5, linestyle='-.')
ax2_twin.set_ylabel('Dilation Size ($d$)', fontsize=12)
ax2_twin.set_ylim(2, 6)
ax2_twin.set_yticks([3, 4, 5])

# 高亮 Semantic Liberation 阶段 (最后三分之一)
ax2.axvspan(T2, MAX_ITERS, color='gray', alpha=0.15, lw=0)

# 添加文字注释 (突出 DDRNet 的特点)
ax2.text(100000, 1.8, 'Semantic\nLiberation', fontsize=12, color='#333333',
         ha='center', va='center', fontweight='bold')

# =====================================================================
# === 4. 图例与排版保存 ===
# =====================================================================

# 提取线条用于全局图例
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]

# 在整张图的正上方添加全局图例
fig.legend(lines, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08), 
           frameon=False, fontsize=12)

# 调整布局，防止文字被裁切
plt.tight_layout()
fig.subplots_adjust(top=0.85) # 给顶部图例留出空间

# 保存高清图片
save_path = 'figure3_architecture_aware_scheduling.pdf'
plt.savefig('work_dirs/figure3_single/figure3_architecture_aware_scheduling.pdf', dpi=300, bbox_inches='tight')
plt.savefig('work_dirs/figure3_single/figure3_architecture_aware_scheduling.png', dpi=300, bbox_inches='tight')

print(f"图片已成功保存为: {save_path} 和 .png 格式")
# plt.show() # 如果在 Jupyter 里可以取消注释直接预览
