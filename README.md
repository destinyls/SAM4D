## Install
### 1. DepthAnything-3
```
conda create -n sam4d python=3.12 -y
conda activate sam4d
# cuda version need to be same with `nvcc --version` for gsplat
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

### 2. SAM3
```
cd sam3
pip install -e .
# For running example notebooks
pip install -e ".[notebooks]"
# For development
pip install -e ".[train,dev]"
```

### 3. SAM3D
```
conda create -n sam3d python=3.11
cd sam-3d-objects
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip3 install torch==2.5.1 torchvision==0.20.1
pip install --no-build-isolation flash_attn==2.8.3
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47
pip install --no-build-isolation git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b

# install sam3d-objects and core dependencies
pip install -e '.[dev]'
pip install -e '.[p3d]' --no-deps # pytorch3d dependency on pytorch is broken, this 2-step approach solves it

# for inference
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
```

## Data Preparation

```
data
|--- videos
|---|--- sequence_0000
|---|---|--- image_0000.png
|---|---|--- image_0001.png
|---|---|--- ...
|---|--- ...
|--- output_sam3_tracking
|---|--- sequence_0000
|---|---|--- masks
|---|---|--- results.json
|---|--- ...
|--- output_sam3d_gaussian
|---|--- sequence_0000
|---|---|--- image_0000
|---|---|---|--- track_000000
|---|---|---|---|--- 000000.gif
|---|---|---|---|--- 000000.ply
|---|--- ...
|--- output_da3_depth
|---|--- sequence_0000
|---|---|--- depth
|---|---|--- visualization  
|---|---|--- ...

```

## Getting Start

### SAM3
```
cd sam3
python sam3_video_predictor_v2.py --sequence_id sequence_0000
```

### SAM3D
```
export HF_HOME=/data2/yanglei/CACHE/huggingface
export TORCH_HOME=/data2/yanglei/CACHE/torch
```  
```
cd sam-3d-objects
python sam3d_inference.py --sequence_id sequence_0000
```
### DepthAnything-3
```
cd Depth-Anything-3
python da3_inference.py --sequence sequence_0000
```

