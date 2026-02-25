查看驱动版本：nvidia-smi

查看 CUDA 运行库版本，可使用 nvcc --version（需安装 CUDA Toolkit）

查看 Linux 版本: lsb_release -a

查看某个文件夹大小：
du -sh /home/tstone10/AUTONOMOUS_DRIVING

df -h

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

github 账号：
1225601453@qq.com
Bbpp1234567890$

# 安装 CUDA Toolkit 11.8:
Ubuntu 24.04 默认编译器是 GCC 13，而 CUDA 11.8 仅支持 GCC 11。因此必须手动安装旧版编译器并进行链接。
sudo apt install gcc-11 g++-11
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --override --silent
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ln -s /usr/bin/gcc-11 /usr/local/cuda-11.8/bin/gcc
sudo ln -s /usr/bin/g++-11 /usr/local/cuda-11.8/bin/g++

nvcc -V

# 解压文件
tar -zxvf gts.tar.gz

# 克隆环境
conda create --name $TARGET_ENV --clone $SOURCE_ENV

# 激活新环境
conda activate $TARGET_ENV

# 删除环境
conda remove -n 环境名  

# 下载 miniconda3 
https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/

wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh
bash ./Miniconda3-py38_22.11.1-1-Linux-x86_64.sh

bash ./Miniconda3-py310_22.11.1-1-Linux-x86_64.sh -b -p ~/miniconda3_py310

# 切换到miniconda3环境 python 3.8
source ~/miniconda3/bin/activate 

# 切换到miniconda3 python 3.10 DFormer
source ~/miniconda3_py310/bin/activate 

# 切换到miniconda3 python 3.9 MambaIR
source ~/miniconda3_py39/bin/activate 

# 切换到miniconda3 python 3.8 TSP6K
source ~/miniconda3_py38/bin/activate 

# 切换到anaconda3环境
source ~/anaconda3/bin/activate

# LRFormer
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# DFormer
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# CUDA 11.8
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

pip install tqdm opencv-python scipy tensorboardX tabulate easydict ftfy regex

# 安装 MMCV
https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html#install-mmcv
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.2/index.html

# 安装和实验 mmsegmentation
CU118 上跑mmsegmentation 似乎只能用 MMCV==2.1.0

https://mmsegmentation.readthedocs.io/en/latest/get_started.html


python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg

# 报错：
CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero.

# 解决：
方案一：最快尝试 —— 加载内核模块（“神技”）
sudo modprobe nvidia-uvm

# 下载数据集
# install OpenXLab CLI tools
pip install -U openxlab
# log in OpenXLab
openxlab login

Access Key ID
z2o3ge4j3rrlnqvpnpwo
Secret Access Key
bng2pzyvlymq0a8rjwagnz10k6ew3w41jrk7dkzv

# 数据集：
https://openxlab.org.cn/datasets/OpenDataLab/ADE20K_2016/tree/main/raw

# KITTI 数据集：
openxlab dataset download --dataset-repo OpenDataLab/KITTI_Object --source-path /raw/data_object_image_2.zip
openxlab dataset download --dataset-repo OpenDataLab/KITTI_Object --source-path /raw/data_object_label_2.zip
openxlab dataset download --dataset-repo OpenDataLab/KITTI_Object --source-path /raw/data_object_calib.zip

# 语法：ps -ef | grep <关键词>
ps -ef | grep python


# 预训练模型下载
https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/model_zoo.md


# 在 CU118 上安装 CU113 的torch
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# TSP6K 训练
python tools/train.py configs/tsp6k/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads.py

# 后台运行，防止训练中断
# new -s 后面跟名字，方便你记住是哪个任务
tmux new -s train_v1

# 查看有哪些正在运行的会话
tmux ls

# 回到之前的会话
tmux attach -t train_v1

# MambaIR 安装
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 数据集下载
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip

# 跑MambaIR
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 basicsr/train.py -opt options/train/mambairv2/train_MambaIRv2_lightSR_x2.yml --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 basicsr/train.py -opt options/train/mambairv2/train_MambaIRv2_lightSR_x4.yml --launcher pytorch

nvcc -V
# 应该显示 11.0
use_cuda118
nvcc -V
# 应该显示 11.8
use_cuda110
nvcc -V
# 变回 11.0

# tensorboard 查看训练过程
tensorboard --logdir=/home/tstone10/AUTONOMOUS_DRIVING/MambaIR/tb_logger --port=6006


# PIDNet
use_cuda118
nvcc --version
source ~/miniconda3_py38/bin/activate 
conda activate openmmlab
python tools/train.py configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py
python tools/train.py configs/pidnet/pidnet-m_2xb6-120k_1024x1024-cityscapes.py
python tools/train.py configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py
# 添加了 ghost conv
python tools/train.py configs/pidnet/pidnet-improved_ghost_conv_s_2xb6-120k_1024x1024-cityscapes.py
# 添加了权重参数 ghost conv & class_weight 训练
python tools/train.py configs/pidnet/pidnet-improved_ghost_conv_class_weight_s_2xb6-120k_1024x1024-cityscapes.py

# DDRNet
python tools/train.py configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py

# TSP6K
python tools/train.py configs/pidnet/pidnet-s_2xb6-120k_1024x1024-tsp6k.py
python tools/train.py configs/pidnet/pidnet-s_2xb6-120k_1024x1024-tsp6k-2.py


# PIDNet 计算 FPS 速度
python tools/analysis_tools/benchmark.py configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py /work_dirs/pidnet-l_2xb6-120k_1024x1024-cityscapes/last_checkpoint
python tools/analysis_tools/benchmark.py configs/pidnet/pidnet-m_2xb6-120k_1024x1024-cityscapes.py /work_dirs/pidnet-m_2xb6-120k_1024x1024-cityscapes/last_checkpoint
python tools/analysis_tools/benchmark.py configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py /work_dirs/pidnet-s_2xb6-120k_1024x1024-cityscapes/last_checkpoint

Done image [50 / 200], fps: 58.68 img / s
Done image [100/ 200], fps: 58.61 img / s
Done image [150/ 200], fps: 58.48 img / s
Done image [200/ 200], fps: 58.61 img / s
Overall fps: 58.61 img / s

python tools/analysis_tools/benchmark.py configs/pidnet/pidnet-improved_s_2xb6-120k_1024x1024-cityscapes.py /work_dirs/pidnet-improved_s_2xb6-120k_1024x1024-cityscape/last_checkpoint

Done image [50 / 200], fps: 58.64 img / s
Done image [100/ 200], fps: 58.81 img / s
Done image [150/ 200], fps: 58.91 img / s
Done image [200/ 200], fps: 58.97 img / s
Overall fps: 58.97 img / s

python tools/analysis_tools/benchmark.py configs/pidnet/pidnet-improved_stem_dilated_conv_s_2xb6-120k_1024x1024-cityscapes.py /work_dirs/pidnet-improved_stem_dilated_conv_s_2xb6-120k_1024x1024-cityscapes/last_checkpoint
[50 / 200], fps: 55.39 img / s
Done image [100/ 200], fps: 56.24 img / s
Done image [150/ 200], fps: 56.33 img / s
Done image [200/ 200], fps: 56.53 img / s
Overall fps: 56.53 img / s

python tools/analysis_tools/benchmark.py configs/pidnet/pidnet-improved_ghost_conv_s_2xb6-120k_1024x1024-cityscapes.py /work_dirs/pidnet-improved_ghost_conv_s_2xb6-120k_1024x1024-cityscapes/last_checkpoint
Done image [50 / 200], fps: 59.61 img / s
Done image [100/ 200], fps: 59.79 img / s
Done image [150/ 200], fps: 59.07 img / s
Done image [200/ 200], fps: 59.30 img / s
Overall fps: 59.30 img / s

# 总结：加入 GhostConv 后，从0开始训练120000次，mIou接近 74%，几乎与PIDNet-s相当 训练24万次，mIou=77.6400
# 多尺度 mIoU: 79.7700
fps: 59.30 img / s
Flops: 40.916G
Params: 5.704M

python tools/analysis_tools/benchmark.py configs/pidnet/pidnet-s_2xb6-120k_1024x1024-tsp6k.py /work_dirs/pidnet-s_2xb6-120k_1024x1024-tsp6k/last_checkpoint

python demo/image_demo.py /home/tstone10/AUTONOMOUS_DRIVING/mmsegmentation/data/TSP6K/image/traffic_00001.jpg configs/pidnet/pidnet-s_2xb6-120k_1024x1024-tsp6k.py /home/tstone10/AUTONOMOUS_DRIVING/mmsegmentation/work_dirs/pidnet-s_2xb6-120k_1024x1024-tsp6k/iter_160000.pth --device cuda:0 --out-file result1.jpg

python test.py \
configs/pidnet/pidnet-s_2xb6-120k_1024x1024-tsp6k.py \
/work_dirs/pidnet-s_2xb6-120k_1024x1024-tsp6k/last_checkpoint \
--img-dir demo/demo.png \  # MMSeg自带的示例图片
--show-dir work_dirs/demo_result \
--opacity 0.5

# 多尺度test
 python tools/test.py configs/pidnet/pidnet-improved_ghost_conv_class_weight_I_only_s_2xb6-120k_1024x1024-cityscapes.py work_dirs/pidnet-improved_ghost_conv_class_weight_I_only_s_2xb6-120k_1024x1024-cityscapes/iter_120000.pth --tta


python tools/analysis_tools/benchmark.py configs/pidnet/pidnet-s_improved_attention_2xb6-120k_1024x1024-cityscapes.py /work_dirs/pidnet-s_improved_attention_2xb6-120k_1024x1024-cityscapes/last_checkpoint

Done image [50 / 200], fps: 41.79 img / s
Done image [100/ 200], fps: 41.74 img / s
Done image [150/ 200], fps: 41.79 img / s
Done image [200/ 200], fps: 41.84 img / s
Overall fps: 41.84 img / s

Average fps of 1 evaluations: 41.84

# PIDNet 计算 FLOPs (计算量) 和 Params (参数量)
python tools/analysis_tools/get_flops.py configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py --shape 1024 2048
Input shape: (1024, 2048)
Flops: 47.517G
Params: 7.721M

python tools/analysis_tools/get_flops.py configs/pidnet/pidnet-m_2xb6-120k_1024x1024-cityscapes.py --shape 1024 2048
python tools/analysis_tools/get_flops.py configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py --shape 1024 2048

python tools/analysis_tools/get_flops.py configs/pidnet/pidnet-improved_s_2xb6-120k_1024x1024-cityscapes.py --shape 1024 2048
Input shape: (1024, 2048)
Flops: 47.522G
Params: 7.73M

python tools/analysis_tools/get_flops.py configs/pidnet/pidnet-s_improved_attention_2xb6-120k_1024x1024-cityscapes.py --shape 1024 2048
Input shape: (1024, 2048)
Flops: 47.53G
Params: 7.767M

python tools/analysis_tools/get_flops.py configs/pidnet/pidnet-improved_ghost_conv_s_2xb6-120k_1024x1024-cityscapes.py --shape 1024 2048
Flops: 40.916G
Params: 5.704M

python tools/analysis_tools/get_flops.py configs/pidnet/pidnet-improved_ghost_conv_se_layer_s_2xb6-120k_1024x1024-cityscapes
Flops: 40.925G
Params: 5.765M

# MMDeploy 安装
https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/get_started.md
# 3.2 install ONNX Runtime
# you can install one to install according whether you need gpu inference
# 3.2.1 onnxruntime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH

python tools/deploy.py \
    configs/mmseg/segmentation_ncnn_static-512x512.py \
    /home/tstone10/AUTONOMOUS_DRIVING/mmsegmentation/configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py \
    /home/tstone10/AUTONOMOUS_DRIVING/mmsegmentation/work_dirs/pidnet-s_2xb6-120k_1024x1024-cityscapes/iter_120000.pth \
    /home/tstone10/AUTONOMOUS_DRIVING/mmsegmentation/demo/demo.png \
    --work-dir work_dirs/pidnet_ncnn \
    --device cpu \
    --dump-info

python tools/deploy.py \
    configs/mmseg/segmentation_onnxruntime_dynamic.py \
    unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py \
    fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth \
    demo/resources/cityscapes.png \
    --work-dir mmdeploy_models/mmseg/ort \
    --device cpu \
    --show \
    --dump-info

# https://www.nuscenes.org/download
1225601453@qq.com
Bbpp1234567890$

 # 3D 分割 pointpillars
 wget https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth

 python demo/pcd_demo.py \
    demo/data/kitti/000008.bin \
    configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
    checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth


python demo/mono_det_demo.py \
    demo/data/kitti/000008.png \
    demo/data/kitti/000008.pkl \
    configs/smoke/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py \
    checkpoints/smoke_dla34_kitti.pth \
    --out-dir smoke_demo_output

# 训练一个mini nuscences    
use_cuda118
conda activate openocc
python tools/train.py configs/bevdet_occ/train_mini.py    

python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir /home/tstone10/AUTONOMOUS_DRIVING/Gen_Video_Img/Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"

python generate.py --task ti2v-5B --size 832*480 --frame_num 33 --sample_steps 20 --ckpt_dir /home/tstone10/AUTONOMOUS_DRIVING/Gen_Video_Img/Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "「镜头：中景固定镜头，古风大殿场景（木质梁柱、竹简案几、武将旗帜，案几放竹简、毛笔、现代财税小册子及算盘，简洁无杂乱），4K画质，写实风格，暖光均匀，光影过渡自然。
角色设定（三个虚拟数字人，非真实肖像，三国古风统一）：
1. 曹操：45-50岁，身材魁梧，威严沉稳，类似陈建斌版曹操气质，深色古风长袍+冠冕，表情严肃带傲娇，语速中等偏慢、语气威严中带调侃，动作：背手、抬手比划、拍案、瞪眼，口型适配台词；
2. 诸葛亮：40岁左右，身形儒雅，浅色长衫+羽扇，表情温和带无奈，语速平缓、语气沉稳带吐槽，动作：摇羽扇、扶额、点头，口型适配台词；
3. 张飞：40岁左右，身材粗壮，武将铠甲，表情憨厚滑稽，语速快、声音洪亮、语气耿直带傻气，动作：叉腰、挠头、拍大腿、举手，口型适配台词。
核心需求：生成三人“丞相学降企业所得税”搞笑对话视频，台词包含财税知识点，中英文无混搭，角色互动自然、表情生动，突出反差萌，可直接导入剪映剪辑，画面纯净无水印。"

# ComfyUI + Wan2.2 5B
diffusion_models / wan2.2_ti2v_5B_fp16.safetensors
https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors

text_encoders / umt5_xxl_fp8_e4m3fn_scaled.safetensors
https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors

vae / wan2.2_vae.safetensors
https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors


# ComfyUI + Wan2.2
use_cuda118
source ~/miniconda3/bin/activate  python 3.11
conda activate comfy
cd Gen_Video_Img/ComfyUI/
python main.py --listen 0.0.0.0 --port 8188

http://19.244.69.121:8188/

# 查看配置文件最终内容
python tools/misc/print_config.py configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py

# 跑 SMOKE + KITTI
python tools/train.py configs/smoke/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py

python tools/test.py \
    configs/smoke/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py \
    work_dirs/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d/epoch_72.pth \
    --show-dir results_vis \
    --task mono_det

# 显示文件大小 GB
ls -lh data/dair-v2x-i/single-infrastructure-side-velodyne.zip 

# 跑 BEVHeight batchSize=4, 接近 15G GPU显存占用
python exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py --amp_backend native -b 4 --gpus=1

# 知识蒸馏
python tools/train.py configs/pidnet/pidnet-improved_ghost_conv_distill_teacher_PIDNet-M-class_weight_s_2xb6-120k_1024x1024-cityscapes.py