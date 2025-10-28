#!/usr/bin/env python3

"""
Cell tracking visualization and trajectory analysis
Creates animated visualizations of tracked cells with intensity measurements
"""

import os
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import tifffile
from scipy.ndimage import zoom
import sys
from pathlib import Path
import json


def load_and_validate_data(original_path, tracked_masks_path):
    """Load and validate input data"""
    try:
        print(f"Loading original images: {original_path}")
        original_stack = tifffile.imread(original_path)  # shape: (T, C, H, W)
        print(f"Original stack shape: {original_stack.shape}")
        
        print(f"Loading tracked masks: {tracked_masks_path}")
        tracked_masks = tifffile.imread(tracked_masks_path)  # shape: (T, H, W)
        print(f"Tracked masks shape: {tracked_masks.shape}")
        
        # Validate dimensions
        if len(original_stack.shape) != 4:
            print("Error: Original stack must be 4D (T, C, H, W)")
            return None, None
        
        if len(tracked_masks.shape) != 3:
            print("Error: Tracked masks must be 3D (T, H, W)")
            return None, None
            
        if original_stack.shape[0] != tracked_masks.shape[0]:
            print(f"Error: Time dimension mismatch - Images: {original_stack.shape[0]}, Masks: {tracked_masks.shape[0]}")
            return None, None
            
        return original_stack, tracked_masks
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def resize_images_to_masks(original_stack, tracked_masks):
    """Resize images to match mask dimensions"""
    target_h, target_w = tracked_masks.shape[1], tracked_masks.shape[2]
    current_h, current_w = original_stack.shape[2], original_stack.shape[3]
    
    if (current_h == target_h) and (current_w == target_w):
        print("✓ Image and mask dimensions already match")
        return original_stack
        
    print(f"Resizing images from {current_h}×{current_w} to {target_h}×{target_w}")
    
    # Calculate zoom factors: (T, C, H, W)
    zoom_factors = (1, 1, target_h / current_h, target_w / current_w)
    resized_original = zoom(original_stack, zoom_factors, order=1)
    
    print(f"✓ Images resized to: {resized_original.shape}")
    return resized_original


def normalize_channels(resized_original):
    """Normalize each channel to [0, 1]"""
    print("Normalizing channels...")
    norm_resized = np.zeros_like(resized_original, dtype=np.float32)
    
    for c in range(resized_original.shape[1]):
        ch = resized_original[:, c]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            norm_resized[:, c] = (ch - ch_min) / (ch_max - ch_min)
        else:
            norm_resized[:, c] = ch  # All values are the same
        print(f"  Channel {c}: [{ch_min:.1f}, {ch_max:.1f}] → [0, 1]")
    
    return norm_resized


def extract_trajectories_and_intensities(norm_resized, tracked_masks, channels_config):
    """Extract cell trajectories and intensity measurements"""
    print("Extracting trajectories and intensities...")
    
    trajectories = {}
    intensities = {}
    
    # Initialize intensity containers for each analysis channel
    for channel_name, channel_idx in channels_config.items():
        if channel_idx is not None:
            intensities[channel_name] = {}
    
    n_frames = tracked_masks.shape[0]
    unique_labels = set()
    
    for t in range(n_frames):
        mask = tracked_masks[t]
        label_ids = np.unique(mask)
        label_ids = label_ids[label_ids != 0]  # Exclude background
        unique_labels.update(label_ids)
        
        for label_id in label_ids:
            yx = np.argwhere(mask == label_id)
            if len(yx) > 0:
                y_mean, x_mean = yx.mean(axis=0)
                
                # Store trajectory
                trajectories.setdefault(label_id, []).append((t, x_mean, y_mean))
                
                # Extract intensities for each channel
                for channel_name, channel_idx in channels_config.items():
                    if channel_idx is not None and channel_idx < norm_resized.shape[1]:
                        img_channel = norm_resized[t, channel_idx]
                        pixels = img_channel[mask == label_id]
                        mean_intensity = pixels.mean()
                        intensities[channel_name].setdefault(label_id, []).append((t, mean_intensity))
    
    print(f"✓ Extracted data for {len(unique_labels)} unique cells across {n_frames} frames")
    return trajectories, intensities


def create_animation(norm_resized, tracked_masks, trajectories, intensities, 
                    channels_config, output_path, animation_config):
    """Create animated visualization"""
    print("Creating animation...")
    
    channels_to_display = animation_config['channels_to_display']
    primary_channel = animation_config['primary_channel']
    intensity_scale = animation_config['intensity_scale']
    
    # Set up plot
    plt.rcParams['animation.embed_limit'] = 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Initialize images
    h, w = tracked_masks.shape[1], tracked_masks.shape[2]
    img1 = axes[0].imshow(np.zeros((h, w, 3)))
    axes[0].set_title("Original Multi-Channel")
    axes[0].axis('off')
    
    img2 = axes[1].imshow(np.zeros((h, w, 3)))
    axes[1].set_title(f"Tracked Cells (colored by {primary_channel})")
    axes[1].axis('off')
    
    title = fig.suptitle("Frame 0")
    label_texts = []
    
    def update(frame_idx):
        nonlocal label_texts
        
        # Clear previous labels
        for txt in label_texts:
            txt.remove()
        label_texts = []
        
        # Build multi-channel frame
        selected_channels = norm_resized[frame_idx, channels_to_display]
        
        if selected_channels.shape[0] == 1:
            multichan_frame = np.repeat(selected_channels, 3, axis=0).transpose(1, 2, 0)
        elif selected_channels.shape[0] == 2:
            ch0, ch1 = selected_channels
            ch2 = np.zeros_like(ch0)
            multichan_frame = np.stack([ch0, ch1, ch2], axis=-1)
        else:
            multichan_frame = selected_channels[:3].transpose(1, 2, 0)
        
        img1.set_data(multichan_frame)
        
        # Create colorized mask
        mask = tracked_masks[frame_idx]
        label_ids = np.unique(mask)
        label_ids = label_ids[label_ids != 0]
        
        color_mask = np.zeros((*mask.shape, 3), dtype=np.float32)
        
        primary_channel_idx = channels_config[primary_channel]
        if primary_channel_idx is not None:
            primary_img = norm_resized[frame_idx, primary_channel_idx]
        else:
            primary_img = np.zeros_like(mask, dtype=np.float32)
        
        for label_id in label_ids:
            yx = np.argwhere(mask == label_id)
            if len(yx) > 0:
                y_mean, x_mean = yx.mean(axis=0)
                
                # Get intensity for coloring
                if primary_channel_idx is not None:
                    pixels = primary_img[mask == label_id]
                    mean_intensity = pixels.mean()
                    # Color by intensity
                    color = cm.viridis(np.clip(mean_intensity / intensity_scale, 0, 1))[:3]
                else:
                    # Default color if no primary channel
                    color = [1.0, 1.0, 1.0]
                
                color_mask[mask == label_id] = color
                
                # Create label text with intensity info
                label_info = [f"ID: {int(label_id)}"]
                for channel_name, channel_idx in channels_config.items():
                    if channel_idx is not None and label_id in intensities[channel_name]:
                        # Find intensity for this frame
                        channel_intensities = intensities[channel_name][label_id]
                        frame_intensities = [intensity for t, intensity in channel_intensities if t == frame_idx]
                        if frame_intensities:
                            label_info.append(f"{channel_name}: {frame_intensities[0]:.2f}")
                
                label_text = "\n".join(label_info)
                txt = axes[1].text(x_mean, y_mean, label_text,
                                 color='white', fontsize=8, ha='center', va='center',
                                 bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=2))
                label_texts.append(txt)
        
        img2.set_data(color_mask)
        title.set_text(f"Frame {frame_idx}")
        return [img1, img2, title] + label_texts
    
    # Create animation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        anim = FuncAnimation(fig, update, frames=len(tracked_masks), 
                           interval=animation_config['interval'], 
                           blit=False, repeat=False)
    
    # Save animation
    print(f"Saving animation to: {output_path}")
    try:
        anim.save(output_path, writer='html', fps=1000/animation_config['interval'])
        print("✓ Animation saved successfully")
    except Exception as e:
        print(f"Error saving animation: {e}")
        return False
    
    plt.close(fig)
    return True


def save_analysis_data(trajectories, intensities, output_prefix):
    """Save trajectory and intensity data"""
    
    # Save trajectories
    traj_file = f"{output_prefix}_trajectories.json"
    traj_data = {}
    for label_id, coords in trajectories.items():
        traj_data[str(label_id)] = [(int(t), float(x), float(y)) for t, x, y in coords]
    
    with open(traj_file, 'w') as f:
        json.dump(traj_data, f, indent=2)
    print(f"✓ Trajectories saved: {traj_file}")
    
    # Save intensities
    intensity_file = f"{output_prefix}_intensities.json"
    intensity_data = {}
    for channel_name, channel_data in intensities.items():
        intensity_data[channel_name] = {}
        for label_id, values in channel_data.items():
            intensity_data[channel_name][str(label_id)] = [(int(t), float(intensity)) for t, intensity in values]
    
    with open(intensity_file, 'w') as f:
        json.dump(intensity_data, f, indent=2)
    print(f"✓ Intensities saved: {intensity_file}")
    
    return traj_file, intensity_file


def main():
    parser = argparse.ArgumentParser(description='Cell tracking visualization and analysis')
    parser.add_argument('--original', required=True, help='Original image stack (TIFF)')
    parser.add_argument('--tracked_masks', '--masks', required=True, help='Tracked masks (TIFF)')
    parser.add_argument('--prefix', required=True, help='Output prefix for files')
    parser.add_argument('--channels_to_display', nargs='+', type=int, default=[0, 1, 2],
                       help='Channels to display in visualization (max 3)')
    parser.add_argument('--primary_channel', type=int, default=1,
                       help='Primary channel for intensity-based coloring')
    parser.add_argument('--analysis_channels', nargs='*', default=['actin_tubulin:1', 'csfe:0'],
                       help='Channels for intensity analysis (format: name:index)')
    parser.add_argument('--intensity_scale', type=float, default=0.05,
                       help='Scale factor for intensity-based coloring')
    parser.add_argument('--interval', type=int, default=200,
                       help='Animation interval in milliseconds')
    parser.add_argument('--sample_id', default='sample', help='Sample ID for summary')
    
    args = parser.parse_args()
    
    print(f"Cell Tracking Visualization")
    print(f"===========================")
    print(f"Original images: {args.original}")
    print(f"Tracked masks: {args.tracked_masks}")
    print(f"Channels to display: {args.channels_to_display}")
    print(f"Primary channel: {args.primary_channel}")
    print(f"Analysis channels: {args.analysis_channels}")
    
    # Load and validate data
    original_stack, tracked_masks = load_and_validate_data(args.original, args.tracked_masks)
    if original_stack is None or tracked_masks is None:
        sys.exit(1)
    
    # Resize images to match masks
    resized_original = resize_images_to_masks(original_stack, tracked_masks)
    
    # Normalize channels
    norm_resized = normalize_channels(resized_original)
    
    # Parse analysis channels
    channels_config = {}
    for channel_spec in args.analysis_channels:
        if ':' in channel_spec:
            name, idx_str = channel_spec.split(':')
            try:
                idx = int(idx_str)
                if 0 <= idx < norm_resized.shape[1]:
                    channels_config[name] = idx
                else:
                    print(f"Warning: Channel index {idx} out of range (0-{norm_resized.shape[1]-1})")
                    channels_config[name] = None
            except ValueError:
                print(f"Warning: Invalid channel index in '{channel_spec}'")
                channels_config[name] = None
        else:
            print(f"Warning: Invalid channel specification '{channel_spec}'. Use format 'name:index'")
    
    # Add primary channel to config
    primary_channel_name = f"channel_{args.primary_channel}"
    channels_config[primary_channel_name] = args.primary_channel
    
    print(f"Channel configuration: {channels_config}")
    
    # Extract trajectories and intensities
    trajectories, intensities = extract_trajectories_and_intensities(
        norm_resized, tracked_masks, channels_config
    )
    
    # Animation configuration
    animation_config = {
        'channels_to_display': args.channels_to_display[:3],  # Max 3 channels
        'primary_channel': primary_channel_name,
        'intensity_scale': args.intensity_scale,
        'interval': args.interval
    }
    
    # Create animation
    animation_path = f"{args.prefix}_tracking_animation.html"
    success = create_animation(
        norm_resized, tracked_masks, trajectories, intensities,
        channels_config, animation_path, animation_config
    )
    
    if not success:
        print("Failed to create animation")
        sys.exit(1)
    
    # Save analysis data
    traj_file, intensity_file = save_analysis_data(trajectories, intensities, args.prefix)
    
    # Generate summary
    summary_file = f"{args.prefix}_visualization_summary.txt"
    n_cells = len(trajectories)
    n_frames = len(tracked_masks)
    
    summary_info = f"""Tracking Visualization Summary for {args.sample_id}
========================================
Input Files:
- Original images: {args.original}
- Tracked masks: {args.tracked_masks}
- Original stack shape: {original_stack.shape}
- Tracked masks shape: {tracked_masks.shape}

Analysis Results:
- Number of frames: {n_frames}
- Number of tracked cells: {n_cells}
- Channels analyzed: {list(channels_config.keys())}
- Animation frames: {n_frames}

Output Files:
- {args.prefix}_tracking_animation.html
- {args.prefix}_trajectories.json
- {args.prefix}_intensities.json
- {args.prefix}_visualization_summary.txt

Configuration:
- Display channels: {args.channels_to_display}
- Primary channel: {args.primary_channel}
- Intensity scale: {args.intensity_scale}
- Animation interval: {args.interval}ms
"""
    
    with open(summary_file, 'w') as f:
        f.write(summary_info)
    
    print(f"\n" + "="*50)
    print(f"Visualization completed successfully!")
    print(f"Generated {n_frames} animation frames for {n_cells} cells")
    print(f"Check the animation: {animation_path}")
    print("="*50)


if __name__ == "__main__":
    main()