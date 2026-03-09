import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark ONNX with TensorRT (Robust Version)')
    parser.add_argument('onnx_file', help='Path to ONNX file')
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--width', type=int, default=2048)
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    return parser.parse_args()

def build_engine(onnx_file_path, args):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # 显式 batch flag
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)

    if not os.path.exists(onnx_file_path):
        print(f"Error: File {onnx_file_path} not found.")
        return None
        
    print(f"Loading ONNX file: {onnx_file_path}...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 添加 Optimization Profile (解决动态尺寸报错的核心)
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    print(f"Detected input tensor name: {input_name}")

    target_shape = (1, 3, args.height, args.width)
    profile.set_shape(input_name, target_shape, target_shape, target_shape)
    config.add_optimization_profile(profile)

    # 开启 FP16
    if args.fp16:
        if builder.platform_has_fast_fp16:
            print("Enabled FP16 Mode (Fast)!")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("Warning: FP16 not supported, falling back to FP32.")

    # 显存池设置 (兼容写法)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
    except AttributeError:
        config.max_workspace_size = 4 * 1024 * 1024 * 1024

    print("Building TensorRT Engine... (This may take a minute)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Error: Failed to build engine.")
        return None

    return trt.Runtime(logger).deserialize_cuda_engine(serialized_engine)

def main():
    args = parse_args()
    
    # 1. 构建引擎
    engine = build_engine(args.onnx_file, args)
    if not engine:
        return

    context = engine.create_execution_context()
    
    # 【核心修复】设置输入尺寸 (使用索引 0)
    # 许多新版 TRT 推荐使用 set_input_shape，但 set_binding_shape 兼容性更好
    input_idx = 0 
    context.set_binding_shape(input_idx, (1, 3, args.height, args.width))

    # 2. 分配显存 (重写：使用索引遍历，修复 TypeError)
    bindings = []
    d_inputs = []
    d_outputs = []
    
    print("Allocating memory...")
    # 使用 range(engine.num_bindings) 确保我们拿到的是整数索引
    for i in range(engine.num_bindings):
        # 获取该 binding 的尺寸
        shape = context.get_binding_shape(i)
        
        # 计算大小 (Volume * sizeof(float32))
        # 注意：即使是 FP16 推理，IO 通常也是 FP32，除非你手动处理 Cast
        size = trt.volume(shape) * 4 
        
        # 分配 GPU 内存
        dev_mem = cuda.mem_alloc(size)
        bindings.append(int(dev_mem))

        # 区分输入和输出
        if engine.binding_is_input(i):
            print(f" - Input binding {i}: {shape}")
            d_inputs.append(dev_mem)
        else:
            print(f" - Output binding {i}: {shape}")
            d_outputs.append(dev_mem)

    # 创建 CUDA 流
    stream = cuda.Stream()

    # 3. Warmup
    print("Warming up (20 iters)...")
    for _ in range(20):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()

    # 4. 正式测速
    iters = 200
    print(f"Benchmarking for {iters} iterations...")
    
    start_time = time.time()
    for _ in range(iters):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / iters
    fps = 1.0 / avg_time

    print("\n============================================")
    print(f"Model: {args.onnx_file}")
    print(f"Precision: {'FP16' if args.fp16 else 'FP32'}")
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Avg Latency: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print("============================================")

if __name__ == '__main__':
    main()
