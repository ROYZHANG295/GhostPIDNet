from mmengine.config import Config
from mmseg.apis import init_model

# 1. 指定你的 Config 路径
config_path = 'configs/space3net/test-0318-2-pidnet-s-no-pretrained-distance-edge-laplacian-attention-I-branch-zero-warmup_2xb6-120k_1024x1024-cityscapes-runable-weight-class.py' # 改成你的文件路径

# 2. 初始化模型 (只构建结构，不加载权重，速度快)
# device='cpu' 即可
model = init_model(config_path, device='cpu')

# 3. 直接打印
print(model)
