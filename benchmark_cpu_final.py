import argparse
import time
import torch
import torch.nn as nn
from mmengine.config import Config
from mmseg.apis import init_model
from mmseg.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark a model on CPU (Pure Inference)')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--height', type=int, default=1024, help='input height')
    parser.add_argument('--width', type=int, default=2048, help='input width')
    parser.add_argument('--iters', type=int, default=50, help='number of iterations')
    parser.add_argument('--threads', type=int, default=1, help='number of CPU threads (1 for single-core sim)')
    return parser.parse_args()

def replace_syncbn(cfg):
    """
    递归替换 Config 中的 SyncBN 为 BN。
    这是在 CPU 上运行的关键，因为 CPU 不支持 SyncBN。
    """
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            if key == 'norm_cfg' and isinstance(value, dict):
                if value.get('type') == 'SyncBN':
                    print(f"Auto-converting SyncBN to BN in {key}...")
                    value['type'] = 'BN'
            else:
                replace_syncbn(value)
    elif isinstance(cfg, list):
        for item in cfg:
            replace_syncbn(item)
    return cfg

def main():
    args = parse_args()
    register_all_modules()

    # 1. 设置 CPU 线程数
    # 限制为 1 可以模拟边缘设备单核性能，GhostConv 优势最明显
    torch.set_num_threads(args.threads)
    print(f'Running on CPU with {args.threads} threads...')

    # 2. 加载并修改 Config
    cfg = Config.fromfile(args.config)
    cfg = replace_syncbn(cfg)  # 自动替换 SyncBN -> BN

    # 3. 初始化模型 (强制 CPU)
    print(f"Initializing model from {args.config}...")
    try:
        model = init_model(cfg, args.checkpoint, device='cpu')
    except AssertionError:
        # 如果 init_model 内部检查失败，尝试手动构建
        from mmseg.models import build_segmentor
        model = build_segmentor(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        if args.checkpoint:
            from mmengine.runner import load_checkpoint
            load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    model.eval()

    # 4. 构造输入 (Cityscapes 尺寸)
    # 模拟经过预处理后的 Tensor (Batch=1, Channel=3, H, W)
    input_shape = (1, 3, args.height, args.width)
    inputs = torch.randn(input_shape)
    
    # 构造伪造的 MetaInfo (防止 encode_decode 报错)
    batch_img_metas = [{
        'img_shape': (args.height, args.width),
        'ori_shape': (args.height, args.width),
        'pad_shape': (args.height, args.width),
        'scale_factor': (1.0, 1.0),
        'flip': False,
        'flip_direction': None
    }]

    # 5. Warm up (热身：让 CPU 缓存和指令预测稳定)
    print('Warming up (10 iters)...')
    with torch.no_grad():
        for _ in range(10):
            model.encode_decode(inputs, batch_img_metas)

    # 6. 正式测试
    print(f'Benchmarking for {args.iters} iterations...')
    start_time = time.perf_counter()

    with torch.no_grad():
        for i in range(args.iters):
            model.encode_decode(inputs, batch_img_metas)
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{args.iters}...")

    end_time = time.perf_counter()
    
    # 7. 计算结果
    total_time = end_time - start_time
    avg_time = total_time / args.iters
    fps = 1.0 / avg_time

    print(f'\n========================================')
    print(f'Model Config: {args.config}')
    print(f'Input Size:   {args.height}x{args.width}')
    print(f'CPU Threads:  {args.threads}')
    print(f'Avg Latency:  {avg_time*1000:.2f} ms')
    print(f'FPS:          {fps:.2f}')
    print(f'========================================')

if __name__ == '__main__':
    main()
