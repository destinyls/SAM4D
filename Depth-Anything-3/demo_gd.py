import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

# from transformers import AutoModel
from accelerate import infer_auto_device_map, dispatch_model

# import GPUtil
# gpus = GPUtil.getGPUs()
# free_gpu_id = min(range(len(gpus)), key=lambda i: gpus[i].memoryUsed)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("/home/yanglei/Depth-Anything-3/pretrained_model/DepthAnything/DepthAnything/DA3NESTED-GIANT-LARGE")

use_multi_gpu = True
if use_multi_gpu and torch.cuda.device_count() > 1:
    # 获取可用GPU数量
    num_gpus = torch.cuda.device_count()
    print("Using {} GPUs for model parallelism".format(num_gpus))



    max_memory_config = {
        0: "22GiB",  # 限制 GPU 0 只用 12GB，迫使模型后半部分去 GPU 1
        1: "22GiB",  # GPU 1 可以多用点
        3: "22GiB", 
        4: "22GiB", 
        5: "22GiB", 
    }

    # 3. 生成 Device Map
    # no_split_module_classes: 防止切断Transformer层或ResNet块。
    # 如果你知道这个模型的主干块叫什么（比如 'Block', 'EncoderLayer'），最好加上。
    # 如果不知道，先不加也可以，accelerate 会尝试按层切分。
    device_map = infer_auto_device_map(
        model, 
        max_memory=max_memory_config, 
        dtype="float16" # 估算大小时假设的数据类型，这很重要，否则可能估算偏大
    )

    
    # 自动计算设备映射
    # 使用infer_auto_device_map生成设备映射字典
    # device_map = infer_auto_device_map(model)
    print("Generated device map: {}".format(device_map))
    
    # 使用dispatch_model将模型分配到多个GPU上
    model = dispatch_model(model, device_map=device_map)
    print("Model has been dispatched to multiple GPUs")
else:
    # 单卡模式
    model = model.to(device)
    print("Model loaded on {}".format(device))

# model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Load sample images and run inference
# image_paths = [
#     "assets/examples/SOH/000.png",
#     "assets/examples/SOH/010.png"
# ]

# da3nested-giant-large  da3-giant
output_dir = '/home/yanglei/Depth-Anything-3/images/output_v1_giant_large_gs'
os.makedirs(output_dir, exist_ok=True)
path = '/home/yanglei/Depth-Anything-3/images/v4_save'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
files.sort()  # 可替换为其他排序逻辑
image_paths = [os.path.join(path, f) for f in files]
print(f'image num: {len(image_paths)}')

# Run inference
# prediction = model.inference(
#     image=image_paths,
#     process_res=504,
#     process_res_method="upper_bound_resize",
#     export_dir=output_dir, # None,
#     export_format="glb"
# )

prediction = model.inference(
    image=image_paths,
    # extrinsics=extrinsics_array,
    # intrinsics=intrinsics_array,
    export_dir=output_dir,
    export_format="npz-glb-gs_ply-gs_video",
    align_to_input_ext_scale=True,
    infer_gs=True,  # Required for gs_ply and gs_video exports
)

print(f"Depth shape: {prediction.depth.shape}")
print(f"Extrinsics: {prediction.extrinsics.shape if prediction.extrinsics is not None else 'None'}")
print(f"Intrinsics: {prediction.intrinsics.shape if prediction.intrinsics is not None else 'None'}")


# conda env export --name A > environment.yml
