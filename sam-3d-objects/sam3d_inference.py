import os
import json
import imageio
import argparse
import time
from tqdm import tqdm

from inference import Inference, ready_gaussian_for_video_rendering, render_video, load_image, load_single_mask, display_image, make_scene, interactive_visualizer

def process_sample(sample, results_dir, base_output_dir, inference, mask_index=None):
    """Process a single sample"""
    start_time = time.time()
    
    try:
        # Check required fields
        if 'image_path' not in sample:
            print("Error: Sample missing 'image_path' field")
            return False
            
        image_path = sample['image_path']
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found - {image_path}")
            return False
        
        # Get image name as identifier
        image_basename = os.path.basename(image_path)
        image_name = os.path.splitext(image_basename)[0]
        
        # Create output directory with structure: base_output_dir/image_0000
        output_dir = os.path.join(base_output_dir, image_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
        
        # Load image
        image = load_image(image_path)
        print(f"Successfully loaded image: {image_basename}")
        
        # Process each category result
        for category_result in sample['results']:
            category = category_result['category']
            
            # Get masks to process
            masks_info = category_result['results']
            
            # If mask_index is specified, process only that mask
            if mask_index is not None and 0 <= mask_index < len(masks_info):
                masks_info = [masks_info[mask_index]]
            
            for mask_info in masks_info:
                try:
                    track_id = mask_info['track_id']
                    # Convert relative path to absolute path
                    mask_relative_path = mask_info['mask_path']
                    mask_absolute_path = os.path.join(results_dir, mask_relative_path)
                    
                    # Build mask path (according to load_single_mask function requirements)
                    mask_dir = os.path.dirname(mask_absolute_path)
                    mask_idx = int(os.path.basename(mask_absolute_path).split('.')[0])
                    
                    # Load mask
                    print(f"Loading mask: directory={mask_dir}, index={mask_idx}")
                    mask = load_single_mask(mask_dir, index=mask_idx)
                    
                    print(f"Processing image: {image_basename}, category: {category}, Track ID: {track_id}")
                    
                    # Run model
                    model_start_time = time.time()
                    print("Starting model inference...")
                    output = inference(image, mask, seed=42)
                    model_time = time.time() - model_start_time
                    print(f"Model inference completed, time taken: {model_time:.2f} seconds")
                    formatted_track_id = str(track_id)
                    
                    # Create final output directory structure: output_dir/track_000000
                    track_output_dir = os.path.join(output_dir, formatted_track_id)
                    os.makedirs(track_output_dir, exist_ok=True)
                    print(f"Created track directory: {track_output_dir}")
                    
                    # Export gaussian splat (as point cloud)
                    ply_filename = f"{formatted_track_id}.ply"
                    ply_path = os.path.join(track_output_dir, ply_filename)
                    output["gs"].save_ply(ply_path)
                    
                    # Render gaussian splat
                    scene_gs = make_scene(output)
                    scene_gs = ready_gaussian_for_video_rendering(scene_gs)
                    
                    video = render_video(
                        scene_gs,
                        r=1,
                        fov=60,
                        pitch_deg=15,
                        yaw_start_deg=-45,
                        resolution=512,
                    )["color"]
                    
                    # Save video as GIF
                    gif_filename = f"{formatted_track_id}.gif"
                    gif_path = os.path.join(track_output_dir, gif_filename)
                    imageio.mimsave(
                        gif_path,
                        video,
                        format="GIF",
                        duration=1000 / 30,  # Assuming 30fps input MP4
                        loop=0,  # 0 means infinite loop
                    )
                    
                    print(f"Saved: {ply_path}")
                    print(f"Saved: {gif_path}")
                    
                except Exception as e:
                    print(f"Error: Failed to process mask {mask_absolute_path} - {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    except Exception as e:
        print(f"Error: Exception occurred while processing sample - {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Record sample processing time
        elapsed_time = time.time() - start_time
        print(f"Sample processing completed, total time taken: {elapsed_time:.2f} seconds")
    
    return True

def main():
    # Record start time
    overall_start_time = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch process samples from results.json')
    parser.add_argument('--input_dir', type=str, 
                        default='../data/output_sam3_tracking',
                        help='Input data directory containing multiple sequence subdirectories')
    parser.add_argument('--sequence', type=str, 
                        default='sequence_0000',
                        help='Sequence name to process, e.g. "sequence_0000", set to "all" to process all sequences')
    parser.add_argument('--mask_index', type=int, default=None,
                        help='Mask index to process, if None process all masks')
    parser.add_argument('--output_dir', type=str, 
                        default='../data/output_sam3d_gaussian',
                        help='Output directory path')
    parser.add_argument('--model_tag', type=str, default='hf',
                        help='Model tag')
    args = parser.parse_args()
    
    # Print configuration info
    print("\n=== Configuration ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Sequence: {args.sequence}")
    print(f"Mask index: {args.mask_index}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model tag: {args.model_tag}")
    
    # Load model
    print("\n=== Loading Model ===")
    config_path = f"checkpoints/{args.model_tag}/pipeline.yaml"
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found - {config_path}")
        return
        
    model_load_start = time.time()
    try:
        inference = Inference(config_path, compile=False)
        model_load_time = time.time() - model_load_start
        print(f"Model loaded successfully, time taken: {model_load_time:.2f} seconds")
    except Exception as e:
        print(f"Error: Failed to load model - {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine sequences to process
    sequences = []
    if args.sequence == 'all':
        # Iterate through all subdirectories starting with 'sequence_' in input_dir
        if os.path.exists(args.input_dir):
            for item in os.listdir(args.input_dir):
                if item.startswith('sequence_') and os.path.isdir(os.path.join(args.input_dir, item)):
                    sequences.append(item)
            if not sequences:
                print(f"Warning: No sequence directories found in {args.input_dir}")
                return
        else:
            print(f"Error: Input directory {args.input_dir} does not exist")
            return
    else:
        # Process specified single sequence
        sequences.append(args.sequence)
    
    # Process each sequence
    print(f"Starting processing of {len(sequences)} sequences...")
    total_success = 0
    total_failed = 0
    
    for sequence_name in tqdm(sequences):
        print(f"\nProcessing sequence: {sequence_name}")
        
        # Build results.json file path
        results_json_path = os.path.join(args.input_dir, sequence_name, 'results.json')
        
        # Check if results.json exists
        if not os.path.exists(results_json_path):
            print(f"Error: results.json file not found for sequence {sequence_name} - {results_json_path}")
            total_failed += 1
            continue
        
        # Read results.json file
        try:
            with open(results_json_path, 'r') as f:
                data = json.load(f)
            print(f"Successfully read {len(data)} samples")
        except Exception as e:
            print(f"Error: Failed to read results.json file - {str(e)}")
            total_failed += 1
            continue
        
        # Get directory of results.json for building absolute paths to masks
        results_dir = os.path.dirname(os.path.abspath(results_json_path))
        
        # Create output directory for this sequence
        sequence_output_dir = os.path.join(args.output_dir, sequence_name)
        os.makedirs(sequence_output_dir, exist_ok=True)
        
        # Process each sample
        sample_success = 0
        sample_failed = 0
        sequence_sample_start_time = time.time()
        
        print(f"\nStarting processing of {len(data)} samples...")
        for i, sample in enumerate(tqdm(data)):
            print(f"\n=== Processing sample {i+1}/{len(data)} ===")
            if process_sample(sample, results_dir, sequence_output_dir, inference, args.mask_index):
                sample_success += 1
            else:
                sample_failed += 1
            
            # Display progress statistics
            if i > 0 and (i+1) % 5 == 0:
                print(f"\nProgress statistics (sample {i+1}/{len(data)}):")
                print(f"  Current success rate: {sample_success/(i+1)*100:.2f}%")
                
        sequence_sample_time = time.time() - sequence_sample_start_time
        print(f"\n=== Sequence {sequence_name} Processing Complete ===")
        print(f"  Successful samples: {sample_success}")
        print(f"  Failed samples: {sample_failed}")
        print(f"  Sequence sample processing time: {sequence_sample_time:.2f} seconds")
        if len(data) > 0:
            avg_time_per_sample = sequence_sample_time / len(data)
            print(f"  Average time per sample: {avg_time_per_sample:.2f} seconds")
            
        total_success += sample_success
        total_failed += sample_failed
    
    # Calculate total time
    overall_time = time.time() - overall_start_time
    
    print(f"\n=== All Sequences Processing Complete ===")
    print(f"Overall statistics:")
    print(f"  Total successful samples: {total_success}")
    print(f"  Total failed samples: {total_failed}")
    total_processed = total_success + total_failed
    if total_processed > 0:
        print(f"  Overall success rate: {total_success/total_processed*100:.2f}%")
    print(f"  Total run time: {overall_time:.2f} seconds")
    print(f"  End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Provide suggestions if there are failures
    if total_failed > 0:
        print(f"\nNote: {total_failed} samples failed to process, please check logs for details")

if __name__ == "__main__":
    main()
    # Maintain existing notebook functionality
    print("\n=== Running in notebook mode ===")
    print("Using default parameter configuration")
    # Note: notebook mode uses default parameter values from main() function
    # To customize parameters in notebook, directly call the main() function with parameters