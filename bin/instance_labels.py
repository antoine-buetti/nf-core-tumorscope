#!/usr/bin/env python3

"""
Instance labels generation using Voronoi-Otsu labeling
Processes T-cell masks to generate smooth, separated cell instances
Enhanced with unique color generation and vectorized color mapping for better visualization
"""

import argparse
import numpy as np
from pathlib import Path
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyclesperanto_prototype as cle
from skimage.transform import resize
from tqdm import tqdm
import sys


def setup_gpu():
    """Initialize GPU for pyclesperanto"""
    try:
        # Initialize GPU
        cle.select_device("TX")  # Try to select GPU
        print(f"Using GPU device: {cle.get_device()}")
    except:
        print("GPU not available, using CPU")


def generate_unique_colors(segmented_data, background_value=0):
    """Generate unique colors for each unique value in the segmented data."""
    unique_values = np.unique(segmented_data)
    unique_values = unique_values[unique_values != background_value]  # Exclude background value

    color_map = {}
    colors_rgb = []
    
    print(f"Generating unique colors for {len(unique_values)} instances...")
    
    for value in tqdm(unique_values, desc="Generating colors"):
        # Hash the value to generate a unique color
        np.random.seed(int(value))  # Use the value as the seed for reproducibility
        color = np.random.rand(3)  # Generate random RGB values
        color_map[value] = mcolors.to_hex(color)  # Convert to hex color
        colors_rgb.append(color)

    return color_map, colors_rgb


def apply_color_map_vectorized(segmented_values, color_map):
    """Apply color map in a vectorized way."""
    print(f"Applying color map to image of shape: {segmented_values.shape}")
    
    # Ensure the color map keys are integers
    keys = np.array(list(color_map.keys()), dtype=int)

    # Convert hex to RGB [0, 1]
    print("Converting hex colors to RGB...")
    colormap_array = np.array([mcolors.to_rgb(color_map[k]) for k in keys], dtype=np.float32)

    # Build a lookup table
    max_label = segmented_values.max()
    print(f"Building lookup table for {max_label + 1} labels...")
    lut = np.zeros((max_label + 1, 3), dtype=np.float32)
    lut[keys] = colormap_array

    # Apply the color map using LUT
    print("Applying vectorized color mapping...")
    rgb_image = lut[segmented_values]
    
    print("Color mapping completed!")
    return rgb_image


def process_tcell_mask(mask_data, spot_sigma=1, outline_sigma=1):
    """Process T-cell mask using Voronoi-Otsu labeling"""
    print(f"Processing mask with shape: {mask_data.shape}")
    print(f"Spot sigma: {spot_sigma}, Outline sigma: {outline_sigma}")
    
    # Push mask to GPU/device
    mask_gpu = cle.push(mask_data)
    
    # Apply Voronoi-Otsu labeling
    segmented = cle.voronoi_otsu_labeling(
        mask_gpu,
        spot_sigma=spot_sigma,
        outline_sigma=outline_sigma
    )
    
    # Pull back to CPU as numpy array
    segmented_np = cle.pull(segmented)
    
    print(f"Segmentation complete. Labels found: {np.max(segmented_np)}")
    return segmented_np


def create_masked_channels(image, mask):
    """Apply mask to all channels of the image"""
    print(f"Creating masked channels for image shape: {image.shape}")
    print(f"Mask shape for masking: {mask.shape}")
    
    # Handle dimension mismatches - resize mask to match image spatial dimensions
    if len(image.shape) == 3:
        # Multi-channel image (H, W, C) or (C, H, W)
        if image.shape[-1] <= 8:  # Likely (H, W, C) format
            target_shape = image.shape[:2]  # (H, W)
            if mask.shape != target_shape:
                print(f"Resizing mask from {mask.shape} to {target_shape}")
                mask_resized = resize(
                    mask, target_shape, 
                    preserve_range=True, order=0
                )
                # Re-binarize after resize
                mask_resized = (mask_resized > 0.5).astype(mask.dtype)
            else:
                mask_resized = mask
            masked_image = image * mask_resized[..., np.newaxis]
            
        else:  # Likely (C, H, W) format
            target_shape = image.shape[1:]  # (H, W)
            if mask.shape != target_shape:
                print(f"Resizing mask from {mask.shape} to {target_shape}")
                mask_resized = resize(
                    mask, target_shape, 
                    preserve_range=True, order=0
                )
                # Re-binarize after resize
                mask_resized = (mask_resized > 0.5).astype(mask.dtype)
            else:
                mask_resized = mask
                
            # Apply mask to each channel
            masked_image = np.zeros_like(image)
            for c in range(image.shape[0]):
                masked_image[c] = image[c] * mask_resized
                
    else:
        # Single channel image (H, W)
        if mask.shape != image.shape:
            print(f"Resizing mask from {mask.shape} to {image.shape}")
            mask_resized = resize(
                mask, image.shape, 
                preserve_range=True, order=0
            )
            # Re-binarize after resize
            mask_resized = (mask_resized > 0.5).astype(mask.dtype)
        else:
            mask_resized = mask
        masked_image = image * mask_resized
    
    return masked_image


def create_visualization(original_mask, segmented_mask, colored_segmented=None, output_path=None, use_unique_colors=True):
    """Create comprehensive visualization comparing original and segmented masks"""
    num_plots = 4 if colored_segmented is not None else 3
    fig, axs = plt.subplots(1, num_plots, figsize=(5*num_plots, 7))
    
    # Original mask
    axs[0].imshow(original_mask, cmap='gray')
    axs[0].set_title('Original Mask')
    axs[0].axis('off')
    
    if use_unique_colors and np.max(segmented_mask) > 0:
        # Generate unique colors for segmented data
        color_map, colors_rgb = generate_unique_colors(segmented_mask)
        
        # Display the color map information
        print("\nGenerated color map:")
        for value, color in list(color_map.items())[:10]:  # Show first 10
            print(f"  Instance {value}: {color}")
        if len(color_map) > 10:
            print(f"  ... and {len(color_map) - 10} more instances")
        
        # Create custom colormap for visualization
        unique_values = np.unique(segmented_mask)
        unique_values = unique_values[unique_values != 0]  # Exclude background
        
        if len(unique_values) > 0:
            # Create a custom colormap
            colors_for_map = ['black']  # Background color
            colors_for_map.extend(colors_rgb)
            custom_cmap = mcolors.ListedColormap(colors_for_map)
            
            # Segmented result with unique colors
            im1 = axs[1].imshow(segmented_mask, cmap=custom_cmap, vmax=len(colors_rgb))
            axs[1].set_title(f'Unique Colors ({np.max(segmented_mask)} instances)')
        else:
            im1 = axs[1].imshow(segmented_mask, cmap='gray')
            axs[1].set_title('Segmented (No instances found)')
    else:
        # Use default colormap
        im1 = axs[1].imshow(segmented_mask, cmap='tab20')
        axs[1].set_title(f'Default Colors ({np.max(segmented_mask)} instances)')
    
    axs[1].axis('off')
    
    # Segmented result with standard colormap for comparison
    im2 = axs[2].imshow(segmented_mask, cmap='tab20')
    axs[2].set_title(f'Standard Colormap ({np.max(segmented_mask)} instances)')
    axs[2].axis('off')
    
    # Show vectorized colored version if available
    if colored_segmented is not None:
        axs[3].imshow(colored_segmented)
        axs[3].set_title('Vectorized Color Mapping')
        axs[3].axis('off')
    
    # Add colorbars
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()


def save_color_map_info(segmented_mask, color_map, output_path):
    """Save color map information to a text file"""
    if np.max(segmented_mask) > 0 and color_map:
        with open(output_path, 'w') as f:
            f.write("Instance Color Mapping\n")
            f.write("=====================\n\n")
            f.write(f"Total instances: {len(color_map)}\n")
            f.write(f"Background value: 0 (black)\n\n")
            
            f.write("Instance ID -> Hex Color -> RGB Values\n")
            f.write("-" * 50 + "\n")
            for value, color in sorted(color_map.items()):
                rgb = mcolors.to_rgb(color)
                rgb_str = f"({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})"
                f.write(f"Instance {value:3d}: {color} -> {rgb_str}\n")
        
        print(f"Color map information saved to: {output_path}")
    else:
        print("No instances found, skipping color map file creation")


def main():
    parser = argparse.ArgumentParser(
        description='Generate instance labels using Voronoi-Otsu segmentation with unique color visualization and vectorized color mapping'
    )
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--mask', required=True, help='Input mask path')
    parser.add_argument('--prefix', required=True, help='Output prefix')
    parser.add_argument(
        '--spot_sigma', type=float, default=1.0, 
        help='Spot sigma for Voronoi-Otsu (default: 1.0)'
    )
    parser.add_argument(
        '--outline_sigma', type=float, default=1.0, 
        help='Outline sigma for Voronoi-Otsu (default: 1.0)'
    )
    parser.add_argument(
        '--frame_index', type=int, default=1,
        help='Frame index for time series (default: 1)'
    )
    parser.add_argument(
        '--no_unique_colors', action='store_true',
        help='Use standard colormap instead of unique colors for visualization'
    )
    parser.add_argument(
        '--save_color_map', action='store_true',
        help='Save color mapping information to a text file'
    )
    parser.add_argument(
        '--save_colored_image', action='store_true',
        help='Save vectorized colored segmentation as RGB TIFF'
    )
    parser.add_argument(
        '--color_output_format', choices=['tiff', 'png'], default='tiff',
        help='Output format for colored image (default: tiff)'
    )
    
    args = parser.parse_args()
    
    print(f"Processing files:")
    print(f"  Image: {args.image}")
    print(f"  Mask: {args.mask}")
    print(f"  Prefix: {args.prefix}")
    print(f"  Unique colors: {not args.no_unique_colors}")
    print(f"  Save colored image: {args.save_colored_image}")
    
    # Setup GPU
    setup_gpu()
    
    try:
        # Load image and mask
        image = imread(args.image)
        mask = imread(args.mask)
        
        print(f"Loaded image shape: {image.shape}")
        print(f"Loaded mask shape: {mask.shape}")
        
        # Handle different image dimensions
        if len(image.shape) > 3:
            # Time series - use specified frame
            frame_idx = min(args.frame_index, image.shape[0] - 1)
            print(f"Using frame {frame_idx} from time series")
            image_frame = image[frame_idx]
            mask_frame = mask[frame_idx] if len(mask.shape) > 2 else mask
        else:
            image_frame = image
            mask_frame = mask
        
        print(f"Image frame shape: {image_frame.shape}")
        print(f"Mask frame shape: {mask_frame.shape}")
        
        # Create masked channels
        masked_channels = create_masked_channels(image_frame, mask_frame)
        
        # For Voronoi-Otsu, use first channel or single channel
        if len(masked_channels.shape) == 3:
            if image_frame.shape[-1] <= 8:  # (H, W, C) format
                processing_channel = masked_channels[..., 0]
                print(f"Using first channel from (H, W, C) format: {processing_channel.shape}")
            else:  # (C, H, W) format
                processing_channel = masked_channels[0]
                print(f"Using first channel from (C, H, W) format: {processing_channel.shape}")
        else:
            processing_channel = masked_channels
            print(f"Using single channel: {processing_channel.shape}")
        
        # Process with Voronoi-Otsu labeling
        segmented_labels = process_tcell_mask(
            processing_channel, 
            spot_sigma=args.spot_sigma,
            outline_sigma=args.outline_sigma
        )
        
        # Generate colored version if requested
        colored_segmented_image = None
        color_map = None
        
        if not args.no_unique_colors and np.max(segmented_labels) > 0:
            print("\nGenerating vectorized colored segmentation...")
            color_map, colors_rgb = generate_unique_colors(segmented_labels)
            colored_segmented_image = apply_color_map_vectorized(segmented_labels, color_map)
        
        # Define output paths
        output_instance_labels = f"{args.prefix}_instance_labels.tiff"
        output_segmented_mask = f"{args.prefix}_segmented_mask.tiff"
        output_masked_channels = f"{args.prefix}_masked_channels.tiff"
        output_overlay = f"{args.prefix}_segmentation_overlay.png"
        output_color_map = f"{args.prefix}_color_map.txt"
        output_colored_image = f"{args.prefix}_colored_segmentation.{args.color_output_format}"
        
        # Save instance labels (the main output)
        imwrite(output_instance_labels, segmented_labels.astype(np.uint16))
        print(f"Instance labels saved: {output_instance_labels}")
        
        # Save segmented mask (binary version)
        segmented_binary = (segmented_labels > 0).astype(np.uint8)
        imwrite(output_segmented_mask, segmented_binary)
        print(f"Segmented mask saved: {output_segmented_mask}")
        
        # Save masked channels
        if len(masked_channels.shape) == 3:
            if image_frame.shape[-1] <= 8:  # (H, W, C) format
                # Transpose to (C, H, W) for TIFF saving
                masked_to_save = np.transpose(masked_channels, (2, 0, 1))
            else:  # Already (C, H, W) format
                masked_to_save = masked_channels
            imwrite(output_masked_channels, masked_to_save.astype(np.uint16))
        else:
            imwrite(output_masked_channels, masked_channels.astype(np.uint16))
        print(f"Masked channels saved: {output_masked_channels}")
        
        # Save colored segmentation image if generated and requested
        if colored_segmented_image is not None and args.save_colored_image:
            if args.color_output_format == 'tiff':
                # Convert to uint8 for TIFF (0-255 range)
                colored_uint8 = (colored_segmented_image * 255).astype(np.uint8)
                # Transpose to (C, H, W) format for TIFF
                colored_to_save = np.transpose(colored_uint8, (2, 0, 1))
                imwrite(output_colored_image, colored_to_save)
            else:  # PNG format
                colored_uint8 = (colored_segmented_image * 255).astype(np.uint8)
                imwrite(output_colored_image, colored_uint8)
            
            print(f"Colored segmentation saved: {output_colored_image}")
        
        # Create comprehensive visualization
        create_visualization(
            processing_channel, 
            segmented_labels,
            colored_segmented_image,
            output_overlay,
            use_unique_colors=not args.no_unique_colors
        )
        
        # Save color map information if requested
        if args.save_color_map and color_map:
            save_color_map_info(segmented_labels, color_map, output_color_map)
        
        print("\nInstance labeling completed successfully!")
        print(f"Summary:")
        print(f"  - Found {np.max(segmented_labels)} instances")
        print(f"  - Generated {len(color_map) if color_map else 0} unique colors")
        print(f"  - Saved {5 + (1 if args.save_colored_image else 0) + (1 if args.save_color_map else 0)} output files")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()