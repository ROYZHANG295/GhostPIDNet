import torch
import torch.nn as nn
from mmseg.apis import init_model
import argparse

# 1. 定义一个包装类，只保留我们需要的部分
# MMSegmentation 的原生 forward 会返回复杂的 DataSample，ONNX 不支持
class PIDNetWrapper(nn.Module):
    def __init__(self, model):
        super(PIDNetWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        # 直接调用 backbone 和 decode_head
        # 绕过 mmseg 的预处理和后处理，只导出核心网络
        feat = self.model.extract_feat(x)
        out = self.model.decode_head.forward(feat)
        
        # PIDNet 的 decode_head 输出可能是个 Tensor
        # 最后我们需要把它上采样回原图大小 (假设输入是 1024x512)
        # 注意：这里为了灵活性，通常导出 1/8 大小的特征图，在外部做 Resize
        # 或者在这里强行 Resize 回输入大小
        out = torch.nn.functional.interpolate(
            out, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return out

def export_onnx(config_path, checkpoint_path, output_file):
    # 2. 加载模型
    print(f"Loading model from {config_path}...")
    model = init_model(config_path, checkpoint_path, device='cpu')
    
    # 3. 包装模型
    wrapped_model = PIDNetWrapper(model)
    wrapped_model.eval()

    # 4. 创建虚拟输入 (根据你的训练尺寸修改，比如 1, 3, 1024, 1024)
    # PIDNet-S 常用尺寸: 1024x1024 或 2048x1024 (Cityscapes)
    # 这里我们用 1, 3, 768, 768 做演示，请根据你的 config 修改！
    input_shape = (1, 3, 512, 1024) 
    dummy_input = torch.randn(input_shape)

    # 5. 导出 ONNX
    print(f"Exporting to {output_file}...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_file,
        input_names=['input'],
        output_names=['output'],
        opset_version=11,  # 推荐 11 或 12，支持 Resize 操作
        do_constant_folding=True,
        # 动态尺寸支持 (可选，如果想固定尺寸则删掉 dynamic_axes)
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        }
    )
    print("Export finished successfully!")

if __name__ == '__main__':
    # 替换为你自己的路径mmsegmentation/configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py
    # CONFIG = 'configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py'
    # CHECKPOINT = 'work_dirs/pidnet-s_2xb6-120k_1024x1024-cityscapes/iter_120000.pth' # 你的权重文件
    # OUTPUT = 'pidnet_s-512-1024.onnx'

    # CONFIG = 'configs/pidnet/pidnet-improved_ghost_conv_class_weight_s_2xb6-120k_1024x1024-cityscapes.py'
    # CHECKPOINT = 'work_dirs/pidnet-improved_ghost_conv_class_weight_s_2xb6-120k_1024x1024-cityscapes/iter_120000.pth' # 你的权重文件
    # OUTPUT = 'ghost_pidnet-512-1024.onnx'

    # CONFIG = 'configs/pidnet/pidnet-faster-pconv-s_2xb6-120k_1024x1024-cityscapes-runable-weight-class.py'
    # CHECKPOINT = 'work_dirs/pidnet-faster-pconv-s_2xb6-120k_1024x1024-cityscapes-runable-weight-class/iter_1000.pth'
    # OUTPUT = 'faster_pidnet-512-1024.onnx'

    # CONFIG = 'configs/pidnet/pidnet-faster-SpaceOptiConv-s_2xb6-120k_1024x1024-cityscapes-runable-weight-class.py'
    # CHECKPOINT = 'work_dirs/pidnet-faster-SpaceOptiConv-s_2xb6-120k_1024x1024-cityscapes-runable-weight-class/iter_1000.pth'
    # OUTPUT = 'spaceOptiConv_pidnet-512-1024.onnx'

    export_onnx(CONFIG, CHECKPOINT, OUTPUT)
