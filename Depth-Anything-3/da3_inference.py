#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth
from accelerate import infer_auto_device_map, dispatch_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('da3_inference')

# 添加命令行参数解析
parser = argparse.ArgumentParser(
    description='Depth Anything 3 Inference Script',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""Example usage:
  python da3_inference.py --sequence sequence_0000
  python da3_inference.py --input_dir /path/to/videos --sequence sequence_0000 --output_dir /path/to/output
  python da3_inference.py --sequence sequence_0000 --process_res 768 --batch_size 4
  """
)
parser.add_argument('--input_dir', type=str, default='../data/videos', help='Input video directory path')
parser.add_argument('--sequence', type=str, required=True, help='Sequence directory name (e.g., sequence_0000)')
parser.add_argument('--output_dir', type=str, default='../data/output_da3_depth', help='Output depth directory path')
parser.add_argument('--model_path', type=str, default='/home/users/ntu/shanhelo/scratch/lei.yang/DepthAnything/DA3NESTED-GIANT-LARGE', help='Path to the model')
parser.add_argument('--process_res', type=int, default=1080, help='Processing resolution')
parser.add_argument('--process_res_method', type=str, default='upper_bound_resize', help='Resize method for processing')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference (limited by GPU memory)')
args = parser.parse_args()

logger.info(f"Command line arguments: {vars(args)}")

# 设置是否使用多卡
use_multi_gpu = True

# 获取设备信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"CUDA devices available: {torch.cuda.device_count()}")
    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
    logger.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# 加载模型
try:
    logger.info(f"Loading model from {args.model_path}")
    start_time = time.time()
    model = DepthAnything3.from_pretrained(args.model_path)
    load_time = time.time() - start_time
    logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# 根据是否使用多卡来处理模型
try:
    if use_multi_gpu and torch.cuda.device_count() > 1:
        # 获取可用GPU数量
        num_gpus = torch.cuda.device_count()
        logger.info(f"Using {num_gpus} GPUs for model parallelism")
        
        # 自动计算设备映射
        start_time = time.time()
        device_map = infer_auto_device_map(model)
        map_time = time.time() - start_time
        logger.info(f"Generated device map in {map_time:.2f} seconds")
        
        # 使用dispatch_model将模型分配到多个GPU上
        start_time = time.time()
        model = dispatch_model(model, device_map=device_map)
        dispatch_time = time.time() - start_time
        logger.info(f"Model has been dispatched to multiple GPUs in {dispatch_time:.2f} seconds")
    else:
        # 单卡模式
        start_time = time.time()
        model = model.to(device)
        move_time = time.time() - start_time
        logger.info(f"Model loaded on {device} in {move_time:.2f} seconds")
    
    model.eval()
    logger.info("Model set to evaluation mode")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise

# 创建输入和输出目录路径
input_sequence_dir = os.path.join(args.input_dir, args.sequence)
output_sequence_dir = os.path.join(args.output_dir, args.sequence)

# 创建输出目录结构
depth_dir = os.path.join(output_sequence_dir, 'depth')
visualization_dir = os.path.join(output_sequence_dir, 'visualization')

try:
    # 创建主序列目录和子目录
    os.makedirs(output_sequence_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    logger.info(f"Output directory structure created: {output_sequence_dir}")
    logger.info(f"  - depth directory: {depth_dir}")
    logger.info(f"  - visualization directory: {visualization_dir}")
except Exception as e:
    logger.error(f"Failed to create output directory structure: {str(e)}")
    raise

# 获取输入目录中的所有图像文件
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
image_paths = []

try:
    if not os.path.exists(input_sequence_dir):
        raise FileNotFoundError(f"Input sequence directory not found: {input_sequence_dir}")
    
    # 获取所有图像文件并按文件名排序
    start_time = time.time()
    for file in sorted(os.listdir(input_sequence_dir)):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(input_sequence_dir, file))
    
    scan_time = time.time() - start_time
    logger.info(f"Found {len(image_paths)} images in {input_sequence_dir} (scan time: {scan_time:.2f}s)")
    
    if not image_paths:
        raise ValueError(f"No images found in {input_sequence_dir}")
except Exception as e:
    logger.error(f"Error processing input directory: {str(e)}")
    raise

# 处理推理
logger.info(f"Running inference on {len(image_paths)} images with process_res={args.process_res}")
logger.info(f"Batch size: {args.batch_size}")

# 如果启用批处理
if args.batch_size > 1:
    logger.info("Processing in batches")
    all_depths = []
    all_extrinsics = None
    all_intrinsics = None
    
    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i+args.batch_size]
        batch_start = time.time()
        
        try:
            batch_prediction = model.inference(
                image=batch_paths,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                export_dir=None,
                export_format="glb"
            )
            
            all_depths.extend(batch_prediction.depth)
            
            # 只保留第一批的相机参数
            if all_extrinsics is None and batch_prediction.extrinsics is not None:
                all_extrinsics = batch_prediction.extrinsics
            if all_intrinsics is None and batch_prediction.intrinsics is not None:
                all_intrinsics = batch_prediction.intrinsics
                
            batch_time = time.time() - batch_start
            logger.info(f"Processed batch {i//args.batch_size + 1}/{(len(image_paths)+args.batch_size-1)//args.batch_size} ({len(batch_paths)} images) in {batch_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing batch {i//args.batch_size + 1}: {str(e)}")
            raise
    
    # 创建预测结果对象
    class Prediction:
        pass
    
    prediction = Prediction()
    prediction.depth = np.array(all_depths)
    prediction.extrinsics = all_extrinsics
    prediction.intrinsics = all_intrinsics
else:
    # 单次处理所有图像
    try:
        start_time = time.time()
        prediction = model.inference(
            image=image_paths,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            export_dir=None,
            export_format="glb"
        )
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        logger.info(f"Average time per image: {inference_time/len(image_paths):.4f} seconds")
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

logger.info(f"Inference results: Depth shape: {prediction.depth.shape}")
if prediction.extrinsics is not None:
    logger.info(f"Extrinsics shape: {prediction.extrinsics.shape}")
if prediction.intrinsics is not None:
    logger.info(f"Intrinsics shape: {prediction.intrinsics.shape}")

# 保存深度图
logger.info(f"Saving depth maps to {output_sequence_dir}...")
save_start_time = time.time()
success_count = 0

for i, (img_path, depth_map) in enumerate(zip(image_paths, prediction.depth)):
    try:
        # 获取原始文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 保存深度图为numpy文件到depth子目录
        depth_npy_path = os.path.join(depth_dir, f"{base_name}_depth.npy")
        np.save(depth_npy_path, depth_map)
        success_count += 1
        
        # 保存可视化的深度图到visualization子目录
        try:
            depth_vis_path = os.path.join(visualization_dir, f"{base_name}_depth.png")
            depth_vis = visualize_depth(depth_map)
            if isinstance(depth_vis, np.ndarray):
                depth_vis_img = Image.fromarray(depth_vis)
                depth_vis_img.save(depth_vis_path)
        except Exception as vis_e:
            logger.warning(f"Failed to save visualization for {base_name}: {str(vis_e)}")
        
        # 每10张图片打印一次进度
        if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
            logger.info(f"Saved {i + 1}/{len(image_paths)} depth maps")
            
    except Exception as e:
        logger.error(f"Error saving depth map for {img_path}: {str(e)}")
        # 继续处理其他图像，不中断整个过程

save_time = time.time() - save_start_time
logger.info(f"Depth map saving completed in {save_time:.2f} seconds")
logger.info(f"Successfully saved {success_count}/{len(image_paths)} depth maps")
logger.info(f"  - Depth files saved to: {depth_dir}")
logger.info(f"  - Visualization files saved to: {visualization_dir}")

# 保存相机参数（如果有）
try:
    if prediction.extrinsics is not None:
        extrinsics_path = os.path.join(output_sequence_dir, "extrinsics.npy")
        np.save(extrinsics_path, prediction.extrinsics)
        logger.info(f"Extrinsics saved to {extrinsics_path}")
    
    if prediction.intrinsics is not None:
        intrinsics_path = os.path.join(output_sequence_dir, "intrinsics.npy")
        np.save(intrinsics_path, prediction.intrinsics)
        logger.info(f"Intrinsics saved to {intrinsics_path}")
except Exception as e:
    logger.error(f"Error saving camera parameters: {str(e)}")

logger.info("Depth Anything 3 inference completed successfully!")

