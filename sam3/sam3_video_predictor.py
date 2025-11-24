import os
import sys
import glob
import json
import torch
import numpy as np
import cv2
import logging
import time
from datetime import datetime
from PIL import Image
import sam3
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    COLORS,
)
from sam3.model_builder import build_sam3_video_predictor
from torchvision.ops import masks_to_boxes
import argparse
from typing import List, Dict, Any, Optional, Tuple, Union

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('sam3_video_predictor')


def load_frame(frame_source: Union[str, np.ndarray]) -> np.ndarray:
    """
    Load a single frame from a file path or numpy array.
    
    Args:
        frame_source: Path to image file or numpy array
        
    Returns:
        Loaded image as numpy array in BGR format for OpenCV processing
    """
    if isinstance(frame_source, str):
        # Load from file
        img = cv2.imread(frame_source)
        if img is None:
            raise ValueError(f"Could not load image from path: {frame_source}")
        return img
    else:
        # Already a numpy array
        # Ensure it's in BGR format if needed
        if frame_source.ndim == 3 and frame_source.shape[2] == 3:
            # Assume input might be in RGB format, convert to BGR
            return cv2.cvtColor(frame_source, cv2.COLOR_RGB2BGR)
        return frame_source


def draw_text(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 0.5,
    thickness: int = 1
) -> None:
    """
    Draw text with a background rectangle on an image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(img, (x, y - text_h - 4), (x + text_w, y + baseline), bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_object_mask_and_box(
    vis_img: np.ndarray,
    obj_id: int,
    binary_mask: Any,
    frame_idx: int
) -> bool:
    """
    Draw mask and bounding box for a single object.
    
    Args:
        vis_img: The image to draw on (modified in-place).
        obj_id: The object ID.
        binary_mask: The binary mask of the object.
        frame_idx: The current frame index.
        
    Returns:
        True if the object was drawn (mask not empty), False otherwise.
    """
    # Extract mask from (prompt, mask) tuple if needed
    if isinstance(binary_mask, tuple) and len(binary_mask) == 2:
        _, mask = binary_mask
    else:
        mask = binary_mask
    
    mask_sum = (
        mask.sum()
        if hasattr(mask, "sum")
        else np.sum(mask)
    )
    
    if mask_sum > 0:
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)

        # Get color
        color_rgb = COLORS[obj_id % len(COLORS)]
        # Convert 0-1 RGB to 0-255 BGR
        color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])
        
        # Draw Mask
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        mask_np = mask_np.astype(bool)
        
        if mask_np.any():
            # Create colored mask
            colored_mask = np.zeros_like(vis_img)
            colored_mask[mask_np] = color_bgr
            
            # Blend
            alpha = 0.5
            # Extract ROI
            roi = vis_img[mask_np]
            overlay = colored_mask[mask_np]
            blended = cv2.addWeighted(roi, 1 - alpha, overlay, alpha, 0)
            vis_img[mask_np] = blended

            # Find bounding box
            box_xyxy = masks_to_boxes(mask.unsqueeze(0)).squeeze()
            # box_xyxy is absolute [x1, y1, x2, y2]
            x1, y1, x2, y2 = box_xyxy.int().tolist()
            
            # Draw BBox
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color_bgr, 2)
            draw_text(vis_img, f"id={obj_id}", (x1, y1 - 5), color=color_bgr, bg_color=(0,0,0))
            
            return True
            
    return False


def visualize_formatted_frame_output(
    frame_idx: int,
    video_frames: List[Union[str, np.ndarray]],
    outputs_list: Union[List[Dict[int, Dict[int, Any]]], Dict[int, Any]],
    titles: Optional[List[str]] = None,
    points_list: Optional[List[Any]] = None,
    points_labels_list: Optional[List[Any]] = None,
    title_suffix: str = "",
    prompt_info: Optional[Dict[str, Any]] = None,
    save: bool = False,
    output_dir: str = ".",
    scenario_name: str = "visualization"
) -> np.ndarray:
    """Visualize up to three sets of segmentation masks on a video frame using OpenCV.

    Args:
        frame_idx: Frame index to visualize
        video_frames: List of image file paths or numpy arrays
        outputs_list: List of {frame_idx: {obj_id: mask_tensor}} or single dict {obj_id: mask_tensor}
        titles: List of titles for each set of outputs_list
        points_list: Optional list of point coordinates
        points_labels_list: Optional list of point labels
        title_suffix: Additional title suffix
        prompt_info: Dictionary with prompt information (boxes, points, etc.)
        save: Whether to save the visualization to file
        output_dir: Base output directory when saving
        scenario_name: Scenario name for organizing saved files
    
    Returns:
        vis_img: The visualized image (numpy array in BGR format)
    """
    # Handle single output dict case
    if isinstance(outputs_list, dict) and frame_idx in outputs_list:
        outputs_list = [outputs_list]
    elif isinstance(outputs_list, dict) and not any(
        isinstance(k, int) for k in outputs_list.keys()
    ):
        single_frame_outputs = {frame_idx: outputs_list}
        outputs_list = [single_frame_outputs]

    num_outputs = len(outputs_list)
    if titles is None:
        titles = [f"Set {i+1}" for i in range(num_outputs)]
    
    # Load image
    img_src = load_frame(video_frames[frame_idx])
    
    # Ensure image is numpy array
    if isinstance(img_src, Image.Image):
        img_src = np.array(img_src)

    # Handle float images (matplotlib imread can return float)
    if (img_src.dtype == np.float32 or img_src.dtype == np.float64) and img_src.max() <= 1.0:
        img_src = (img_src * 255).astype(np.uint8)
    elif img_src.dtype != np.uint8:
        img_src = img_src.astype(np.uint8)
    
    # Convert to BGR for OpenCV
    if img_src.ndim == 3 and img_src.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
    elif img_src.ndim == 3 and img_src.shape[2] == 4:
        # Handle RGBA
        img_bgr = cv2.cvtColor(img_src, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR)

    img_H, img_W = img_bgr.shape[:2]
    
    # Create a list to store visualized images for each output set
    vis_images = []

    for idx in range(num_outputs):
        # Create a copy for drawing
        vis_img = img_bgr.copy()
        outputs_set = outputs_list[idx]
        ax_title = titles[idx]
        
        # Draw title
        title_text = f"Frame {frame_idx} - {ax_title}{title_suffix}"
        draw_text(vis_img, title_text, (10, 30), font_scale=0.7, thickness=2)

        if frame_idx in outputs_set:
            _outputs = outputs_set[frame_idx]
        else:
            _outputs = {}

        # Draw prompts on first frame
        if prompt_info and frame_idx == 0:
            if "boxes" in prompt_info:
                for box in prompt_info["boxes"]:
                    # box is in [x, y, w, h] normalized format
                    x, y, w, h = box
                    # Convert to absolute coordinates
                    x1, y1 = int(x * img_W), int(y * img_H)
                    x2, y2 = int((x + w) * img_W), int((y + h) * img_H)
                    
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow
                    draw_text(vis_img, "PROMPT BOX", (x1, y1 - 5), color=(0, 255, 255), bg_color=(0, 0, 0))

            if "points" in prompt_info and "point_labels" in prompt_info:
                points = np.array(prompt_info["points"])
                labels = np.array(prompt_info["point_labels"])
                # Convert normalized to pixel coordinates
                points_pixel = points * np.array([img_W, img_H])

                for pt, label in zip(points_pixel, labels):
                    px, py = int(pt[0]), int(pt[1])
                    if label == 1:
                        # Positive point - Green star-like (Circle + Cross)
                        cv2.circle(vis_img, (px, py), 5, (0, 255, 0), -1)
                        cv2.drawMarker(vis_img, (px, py), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
                    else:
                        # Negative point - Red star-like
                        cv2.circle(vis_img, (px, py), 5, (0, 0, 255), -1)
                        cv2.drawMarker(vis_img, (px, py), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)

        objects_drawn = 0
            
        for obj_id, binary_mask in _outputs.items():
            if draw_object_mask_and_box(vis_img, obj_id, binary_mask, frame_idx):
                objects_drawn += 1

        if objects_drawn == 0:
            text = "No objects detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = (img_W - text_w) // 2
            y = (img_H + text_h) // 2
            cv2.putText(vis_img, text, (x, y), font, font_scale, (0, 0, 255), thickness)

        # Draw additional points if provided
        if points_list is not None and points_list[idx] is not None:
            pts = points_list[idx]
            lbls = points_labels_list[idx] if points_labels_list else [1]*len(pts)
            for pt, lbl in zip(pts, lbls):
                # Assuming pts are absolute [x, y]
                px, py = int(pt[0]), int(pt[1])
                color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
                cv2.circle(vis_img, (px, py), 5, color, -1)
        
        vis_images.append(vis_img)

    # Concatenate images if multiple outputs
    if len(vis_images) > 1:
        final_img = np.hstack(vis_images)
    else:
        final_img = vis_images[0]

    # Save if requested
    if save:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{scenario_name}_frame_{frame_idx:05d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
        print(f"Visualization saved to {filepath}")

    return final_img

def setup_predictor(
    checkpoint_path: str,
    device: str = "cuda",
    gpus_to_use: Optional[List[int]] = None
) -> Any:
    """
    Initialize the SAM3 video predictor with enhanced error handling and logging.
    """
    try:
        logger.info(f"Setting up predictor with checkpoint: {checkpoint_path}")
        
        # Check if checkpoint exists
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Check device availability
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available, falling back to CPU")
                device = "cpu"
            else:
                logger.info(f"CUDA is available with {torch.cuda.device_count()} devices")
        else:
            logger.info("Using CPU device")
        
        # Configure GPUs to use
        if gpus_to_use is None:
            # Use all available GPUs on the machine by default
            if torch.cuda.is_available():
                gpus_to_use = list(range(torch.cuda.device_count()))
                logger.info(f"Using all available GPUs: {gpus_to_use}")
            else:
                gpus_to_use = []
                logger.info("No GPUs available, using CPU")
        else:
            logger.info(f"Using specified GPUs: {gpus_to_use}")
        
        # Measure initialization time
        init_start = time.time()
        logger.info("Building SAM3 video predictor...")
        
        try:
            predictor = build_sam3_video_predictor(
                checkpoint_path=checkpoint_path, 
                gpus_to_use=gpus_to_use
            )
            init_time = time.time() - init_start
            logger.info(f"Predictor initialized successfully in {init_time:.2f} seconds")
            
            return predictor
        except Exception as e:
            logger.error(f"Failed to build SAM3 video predictor: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error during predictor setup: {str(e)}")
        raise


def load_video_data(video_path: str, max_frames: int = -1) -> List[str]:
    """
    Load video frames from a directory of images according to the specified directory structure.
    
    Args:
        video_path: Path to the directory containing images
        max_frames: Maximum number of frames to load, -1 for all frames
        
    Returns:
        List of image paths sorted by filename
    """
    # Ensure path is a directory
    if not os.path.isdir(video_path):
        raise ValueError(f"Expected directory path, got: {video_path}")
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(video_path, f'*{ext}')))
    
    # Sort the files by name to ensure correct frame order
    image_files.sort()
    
    # Limit the number of frames if specified
    if max_frames > 0:
        image_files = image_files[:max_frames]
    
    if not image_files:
        raise ValueError(f"No image files found in directory: {video_path}")
    
    # Try to sort numerically if possible for better frame ordering
    try:
        image_files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        # Already sorted lexicographically
        pass
    
    return image_files


def initialize_session(predictor: Any, video_path: str) -> int:
    """
    Start a new session with the predictor.
    """
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    return response["session_id"]

def add_prompt(predictor: Any, session_id: int, frame_idx: int, text: str) -> Dict[str, Any]:
    """
    Add a text prompt to a specific frame.
    """
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=text,
        )
    )
    return response["outputs"]

def apply_prompt(predictor: Any, session_id: int, frame_idx: int, text: str) -> Dict[str, Any]:
    """
    Reset session and add a text prompt to a specific frame.
    """
    # Reset session first to ensure clean state for new prompt
    _ = predictor.handle_request(
        request=dict(
            type="reset_session",
            session_id=session_id,
        )
    )

    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=text,
        )
    )
    return response["outputs"]


def propagate_in_video(predictor: Any, session_id: int) -> Dict[int, Dict[str, Any]]:
    """
    Propagate masks from frame 0 to the end of the video.
    """
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


def abs_to_rel_coords(
    coords: List[List[float]],
    IMG_WIDTH: int,
    IMG_HEIGHT: int,
    coord_type: str = "point"
) -> List[List[float]]:
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        IMG_WIDTH: Image width
        IMG_HEIGHT: Image height
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")


def visualize_results(
    video_frames_for_vis: List[Union[str, np.ndarray]],
    outputs_per_frame: Dict[int, Any],
    stride: int = 60,
    save: bool = False,
    output_dir: str = "./visualization_outputs",
    format: str = "jpg",
    fps: int = 10
) -> None:
    """
    Visualize the results at a given stride and optionally save to files or video.
    """
    # Ensure outputs are prepared for visualization
    # Ensure outputs are prepared for visualization
    # Check if the first frame's output is already processed (keys are integers)
    # or if it is raw (has "out_obj_ids")
    if not outputs_per_frame:
        formatted_outputs = {}
    else:
        first_frame_idx = next(iter(outputs_per_frame.keys()))
        first_frame_out = outputs_per_frame[first_frame_idx]
        
        is_raw = False
        if isinstance(first_frame_out, dict) and "out_obj_ids" in first_frame_out:
            is_raw = True
                
        if not is_raw:
            formatted_outputs = outputs_per_frame
        else:
            formatted_outputs = prepare_masks_for_visualization(outputs_per_frame)
    
    # Create output directory if saving is enabled
    if save:
        os.makedirs(output_dir, exist_ok=True)
    
    if format == "jpg":
        # Save individual image frames
        for frame_idx in range(0, len(formatted_outputs), stride):
            visualize_formatted_frame_output(
                frame_idx,
                video_frames_for_vis,
                outputs_list=[formatted_outputs],
                titles=["SAM 3 Dense Tracking outputs"],
                save=save,
                output_dir=output_dir,
                scenario_name="sam3_tracking"
            )
    elif format == "mp4":
        # Generate MP4 video
        generate_video_from_visualizations(
            video_frames_for_vis,
            formatted_outputs,
            output_dir,
            fps=fps,
            stride=stride
        )
    else:
        print(f"Unknown output format: {format}. Using 'jpg' as default.")
        # Fallback to jpg format
        for frame_idx in range(0, len(formatted_outputs), stride):
            visualize_formatted_frame_output(
                frame_idx,
                video_frames_for_vis,
                outputs_list=[formatted_outputs],
                titles=["SAM 3 Dense Tracking outputs"],
                save=save,
                output_dir=output_dir,
                scenario_name="sam3_tracking"
            )

def generate_video_from_visualizations(
    video_frames_for_vis: List[Union[str, np.ndarray]],
    formatted_outputs: Dict[int, Any],
    output_dir: str,
    fps: int = 10,
    stride: int = 1
) -> None:
    """
    Generate an MP4 video from the visualization frames using OpenCV.
    """
    # Determine the size of the output frames
    # Get first frame to determine size
    first_frame_img = visualize_formatted_frame_output(
        0,  # First frame
        video_frames_for_vis,
        outputs_list=[formatted_outputs],
        titles=["SAM 3 Dense Tracking outputs"],
        save=False
    )
    
    height, width = first_frame_img.shape[:2]
    
    # Define the codec and create VideoWriter object
    video_path = os.path.join(output_dir, "sam3_tracking_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    print(f"Generating video: {video_path} (FPS: {fps}, Size: {width}x{height})")
    
    # Process each frame
    for frame_idx in range(0, len(formatted_outputs), stride):
        # Generate the visualization for this frame
        img_array = visualize_formatted_frame_output(
            frame_idx,
            video_frames_for_vis,
            outputs_list=[formatted_outputs],
            titles=["SAM 3 Dense Tracking outputs"],
            save=False
        )
        
        # Ensure the image size matches the video writer
        # Ensure the image size matches the video writer
        if img_array.shape[:2] != (height, width):
             img_array = cv2.resize(img_array, (width, height))
        
        # Convert RGB to BGR for OpenCV VideoWriter


        # Write the frame to the video
        out.write(img_array)
        
    # Release the video writer
    out.release()
    print(f"Video saved successfully to {video_path}")


def save_masks_png(
    video_frames: List[Union[str, np.ndarray]],
    outputs_per_frame: Dict[int, Dict[int, Any]],
    output_dir: str = "./output_sam3_tracking/masks"
) -> Dict[int, Dict[int, str]]:
    """
    Save segmentation masks as PNG files, preserving original RGB in masked areas and setting rest to black.
    
    Args:
        video_frames: List of image file paths or numpy arrays
        outputs_per_frame: Dictionary of {frame_idx: {track_id: mask_tensor}}
        output_dir: Base directory to save PNG files (will create subdirs as masks/image_name/)
        
    Returns:
        Dictionary mapping (frame_idx, track_id) to relative mask file path
    """
    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)
    
    mask_paths = {}
    
    # Ensure outputs are prepared for saving
    if not outputs_per_frame:
        return mask_paths
    
    first_frame_idx = next(iter(outputs_per_frame.keys()))
    first_frame_out = outputs_per_frame[first_frame_idx]
    
    # Check if outputs need formatting
    is_raw = False
    if isinstance(first_frame_out, dict) and "out_obj_ids" in first_frame_out:
        is_raw = True
            
    if not is_raw:
        formatted_outputs = outputs_per_frame
    else:
        formatted_outputs = prepare_masks_for_visualization(outputs_per_frame)
    
    # Save each mask as PNG file
    for frame_idx, frame_outputs in formatted_outputs.items():
        if frame_idx not in mask_paths:
            mask_paths[frame_idx] = {}
        
        # Load the original image for this frame
        img = load_frame(video_frames[frame_idx])
        
        # Ensure image is numpy array in BGR format for OpenCV
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        # Handle float images
        if (img.dtype == np.float32 or img.dtype == np.float64) and img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # Convert RGB to BGR if needed
        if img.ndim == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img.copy()
        
        # Get image name from path or create one
        if isinstance(video_frames[frame_idx], str):
            image_name = os.path.splitext(os.path.basename(video_frames[frame_idx]))[0]
        else:
            image_name = f"frame_{frame_idx:05d}"
        
        # Create image-specific directory
        image_dir = os.path.join(output_dir, image_name)
        os.makedirs(image_dir, exist_ok=True)
        
        for track_id, item in frame_outputs.items():
            # Extract mask from (prompt, mask) tuple if needed
            if isinstance(item, tuple) and len(item) == 2:
                prompt, mask = item
            else:
                mask = item
                prompt = "object"  # Default prompt if not available
                
            # Convert mask to numpy array if needed
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            # Ensure mask is 2D and boolean
            mask_np = mask_np.astype(bool)
            
            # Create output image with black background
            output_img = np.zeros_like(img_bgr)
            
            # Copy original image content where mask is True
            if mask_np.ndim == 2:
                # 2D mask: apply to all channels
                output_img[mask_np] = img_bgr[mask_np]
            elif mask_np.ndim == 3 and mask_np.shape[2] == 1:
                # Single-channel mask: squeeze to 2D
                mask_np = np.squeeze(mask_np)
                output_img[mask_np] = img_bgr[mask_np]
            else:
                # Multi-channel mask: take first channel
                mask_np = mask_np[:, :, 0]
                output_img[mask_np] = img_bgr[mask_np]
            
            # Create filename and save path
            mask_filename = f"{track_id}.png"
            mask_filepath = os.path.join(image_dir, mask_filename)
            
            # Save as PNG
            cv2.imwrite(mask_filepath, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
            
            # Store relative path (from output_dir)
            mask_paths[frame_idx][track_id] = os.path.join("masks", image_name, mask_filename)
    
    return mask_paths


def save_results_json(
    video_frames: List[Union[str, np.ndarray]],
    outputs_per_frame: Dict[int, Dict[int, Any]],
    mask_paths: Dict[int, Dict[int, str]],
    output_dir: str = "./output_sam3_tracking",
) -> str:
    """
    Save detection results in JSON format as specified.
    Format: [{"image_path": "", "results":[{"category":"", "results": [{"track_id":0, "mask_path": "mask_path"}]}]}]
    
    Args:
        video_frames: List of image file paths or numpy arrays
        outputs_per_frame: Dictionary of {frame_idx: {track_id: mask_tensor}}
        mask_paths: Dictionary mapping (frame_idx, track_id) to relative mask file path
        output_dir: Directory to save JSON file
        
    Returns:
        Path to the saved JSON file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare results list in the specified format
    results_list = []
    
    # Ensure outputs are prepared
    if not outputs_per_frame:
        return None
    
    first_frame_idx = next(iter(outputs_per_frame.keys()))
    first_frame_out = outputs_per_frame[first_frame_idx]
    
    # Check if outputs need formatting
    is_raw = False
    if isinstance(first_frame_out, dict) and "out_obj_ids" in first_frame_out:
        is_raw = True
            
    if not is_raw:
        formatted_outputs = outputs_per_frame
    else:
        formatted_outputs = prepare_masks_for_visualization(outputs_per_frame)
    
    # Process each frame
    for frame_idx in sorted(formatted_outputs.keys()):
        # Get image path
        if isinstance(video_frames[frame_idx], str):
            image_path = video_frames[frame_idx]  # Use filename only for better portability
        else:
            # For numpy array frames, create a filename
            image_path = f"frame_{frame_idx:05d}.jpg"
        
        # Prepare frame entry according to the specified format
        frame_entry = {
            "image_path": image_path,
            "results": []
        }
        
        # Group tracks by category - using actual prompt text stored with each mask
        # Create a dictionary to store category entries
        category_entries = {}
        
        # Add tracks to appropriate category results
        for track_id, item in formatted_outputs[frame_idx].items():
            # Get prompt from the (prompt, mask) tuple, fallback to default if not available
            prompt_text = "object"  # Default fallback
            if isinstance(item, tuple) and len(item) == 2:
                prompt_text, _ = item
                
            # Initialize category entry if it doesn't exist
            if prompt_text not in category_entries:
                category_entries[prompt_text] = {
                    "category": prompt_text,
                    "results": []
                }
                
            # Add track entry if mask path exists
            if frame_idx in mask_paths and track_id in mask_paths[frame_idx]:
                track_entry = {
                    "track_id": int(track_id),
                    "mask_path": mask_paths[frame_idx][track_id]
                }
                category_entries[prompt_text]["results"].append(track_entry)
                
        # Add all category entries that have results
        for category_entry in category_entries.values():
            if category_entry["results"]:
                frame_entry["results"].append(category_entry)
        
        # Add frame entry to results list
        results_list.append(frame_entry)
    
    # Save to JSON file
    json_filepath = os.path.join(output_dir, "results.json")
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {json_filepath}")
    print(f"Total frames processed: {len(results_list)}")
    # Calculate total objects
    total_objects = sum(len(category['results']) for frame in results_list for category in frame['results'])
    print(f"Total objects detected: {total_objects}")
    
    return json_filepath


def merge_frame_outputs(
    all_outputs: Dict[int, Dict[int, Any]],
    new_outputs: Dict[int, Dict[str, Any]],
    max_obj_id: int,
    prompt: str,
) -> int:
    """
    Merge new outputs into the accumulated outputs, updating object IDs to avoid collisions.
    
    Args:
        all_outputs: Dictionary accumulating outputs from all prompts.
        new_outputs: Outputs from the current prompt execution.
        max_obj_id: The maximum object ID used so far.
        
    Returns:
        Updated max_obj_id.
    """
    current_run_max_id = -1
    objects_found = 0

    for frame_idx, frame_out in new_outputs.items():
        if frame_idx not in all_outputs:
            all_outputs[frame_idx] = {}

        if "out_obj_ids" in frame_out and "out_binary_masks" in frame_out:
            obj_ids = frame_out["out_obj_ids"]
            masks = frame_out["out_binary_masks"]

            # Convert to list if tensor or numpy array
            if isinstance(obj_ids, torch.Tensor):
                obj_ids = obj_ids.tolist()
            elif isinstance(obj_ids, np.ndarray):
                obj_ids = obj_ids.tolist()

            for idx, obj_id in enumerate(obj_ids):
                mask = masks[idx]

                # Only count/merge if mask is not empty
                if mask.sum() > 0:
                    new_obj_id = obj_id + max_obj_id + 1
                    all_outputs[frame_idx][new_obj_id] = (prompt, mask)

                    if obj_id > current_run_max_id:
                        current_run_max_id = obj_id
                    
                    objects_found += 1
    
    # Update max_obj_id for next iteration
    if current_run_max_id != -1:
        max_obj_id += (current_run_max_id + 1)
        
    return max_obj_id


def run_inference(
    predictor: Any,
    session_id: int,
    prompts: List[str],
    prompt_frame_idx: int = 0
) -> Dict[int, Dict[int, Any]]:
    """
    Run inference for multiple prompts and merge results.
    
    Args:
        predictor: The SAM3 video predictor instance.
        session_id: The current session ID.
        prompts: List of text prompts to process.
        prompt_frame_idx: The frame index to apply prompts to.
        
    Returns:
        Merged outputs for all frames.
    """
    all_outputs_per_frame = {}
    max_obj_id = -1

    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}: '{prompt}' at frame {prompt_frame_idx}...")

        # Reset session
        _ = predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )

        # Add prompt
        _ = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=prompt_frame_idx,
                text=prompt,
            )
        )

        # Propagate
        print(f"Propagating masks for '{prompt}'...")
        outputs_per_frame = propagate_in_video(predictor, session_id)

        # Merge outputs
        max_obj_id = merge_frame_outputs(all_outputs_per_frame, outputs_per_frame, max_obj_id, prompt)

    return all_outputs_per_frame


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="SAM3 Video Predictor")
    parser.add_argument("--checkpoint_path", type=str, default="/data2/yanglei/SAM3/SAM3/sam3/sam3.pt", help="Path to the model checkpoint. If None, downloads from HF.")
    parser.add_argument("--sequence_id", type=str, default="sequence_0000_mini", help="Sequence ID (e.g., sequence_0000)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("--save_results", action="store_true", default=True, help="Whether to save detection results in JSON and PNG format.")
    parser.add_argument("--masks_dir", type=str, default="masks", help="Subdirectory to save mask PNG files.")
    parser.add_argument("--data_root", type=str, default="../data", help="Root directory for data.")
    
    return parser.parse_args()


def main():
    start_time = time.time()
    logger.info("Starting SAM3 video prediction process")
    
    try:
        # Configuration
        args = get_args()

        checkpoint_path = args.checkpoint_path
        sequence_id = args.sequence_id
        device = args.device
        data_root = args.data_root
        
        # Validate required arguments
        if not sequence_id:
            raise ValueError("sequence_id is required")
        
        if not data_root:
            raise ValueError("data_root is required")
        
        # Construct paths based on the directory structure
        video_path = os.path.join(data_root, "videos", sequence_id)
        output_dir = os.path.join(data_root, "output_sam3_tracking", sequence_id)
        masks_dir = os.path.join(output_dir, "masks")
        
        # Validate input directory structure
        if not os.path.isdir(data_root):
            raise ValueError(f"Data root directory does not exist: {data_root}")
        
        if not os.path.isdir(video_path):
            raise ValueError(f"Video directory does not exist for sequence {sequence_id}: {video_path}")
        
        # Create output directory structure if it doesn't exist
        try:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            logger.info(f"Created output directories: {output_dir}, {masks_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directories: {str(e)}")
            raise
        
        logger.info(f"Processing sequence: {sequence_id}")
        logger.info(f"Input directory: {video_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Masks directory: {masks_dir}")

        # Other visualization parameters (not yet argparse-ified)
        vis_stride = 1
        output_format = "jpg"  # Options: "jpg" for image outputs, "mp4" for video output
        save_visualizations = True  # Set to True to save visualization images or video
        video_fps = 10  # FPS for the output video if format is "mp4"

        # 1. Setup
        logger.info("Setting up predictor...")
        try:
            predictor = setup_predictor(
                checkpoint_path=checkpoint_path,
                device=device,
            )
            logger.info(f"Predictor initialized successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {str(e)}")
            raise

        # 2. Load Data
        logger.info(f"Loading video data from {video_path}...")
        try:
            video_frames_for_vis = load_video_data(video_path)
            logger.info(f"Loaded {len(video_frames_for_vis)} frames")
        except Exception as e:
            logger.error(f"Error loading video data: {str(e)}")
            raise

        # 3. Initialize Session
        logger.info("Initializing session...")
        try:
            session_id = initialize_session(predictor, video_path)
            logger.info(f"Session initialized with ID: {session_id}")
        except Exception as e:
            logger.error(f"Failed to initialize session: {str(e)}")
            raise

        prompts = ["car", "bus"]
        logger.info(f"Using prompts: {prompts}")

        # 5. Run Inference
        inference_start = time.time()
        logger.info("Running inference...")
        try:
            all_outputs_per_frame = run_inference(predictor, session_id, prompts)
            inference_time = time.time() - inference_start
            logger.info(f"Inference completed in {inference_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise

        # 6. Visualize Results
        if save_visualizations:
            vis_start = time.time()
            logger.info(f"Visualizing merged results... (Format: {output_format})")
            try:
                visualize_results(
                    video_frames_for_vis, 
                    all_outputs_per_frame, 
                    stride=vis_stride,
                    save=save_visualizations,
                    output_dir=output_dir,
                    format=output_format,
                    fps=video_fps
                )
                vis_time = time.time() - vis_start
                logger.info(f"Visualization completed in {vis_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error during visualization: {str(e)}")
                # Continue even if visualization fails
                logger.warning("Continuing without visualization due to error")
    
        # 7. Save Results (JSON and PNG)
        if args.save_results:
            save_start = time.time()
            logger.info("Saving detection results...")
            try:
                # Create full masks directory path
                masks_full_dir = os.path.join(output_dir, args.masks_dir)
                # Save masks as PNG files
                mask_paths = save_masks_png(
                    video_frames_for_vis,
                    all_outputs_per_frame,
                    output_dir=masks_full_dir
                )
                # Save results as JSON
                json_path = save_results_json(
                    video_frames_for_vis,
                    all_outputs_per_frame,
                    mask_paths,
                    output_dir=output_dir,
                )
                save_time = time.time() - save_start
                logger.info(f"Results saved in {save_time:.2f} seconds")
                logger.info(f"JSON results saved to: {json_path}")
                logger.info(f"Masks saved to: {masks_full_dir}")
            except Exception as e:
                logger.error(f"Error saving results: {str(e)}")
                raise
        
        total_time = time.time() - start_time
        logger.info(f"Processing completed successfully in {total_time:.2f} seconds!")
        
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()