#!/usr/bin/env python3

"""
Cell tracking using Trackastra deep learning model
Standalone script for Nextflow module
"""

import argparse
import torch
import pickle
import sys
from pathlib import Path
from tifffile import imread, imwrite
import skimage as ski
import numpy as np

# Import trackastra components
try:
    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks
except ImportError as e:
    print(f"Error importing trackastra: {e}")
    print("Please ensure trackastra is installed: pip install trackastra")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Cell tracking using Trackastra')
    parser.add_argument('--images', required=True, help='Input time-series images (TIFF)')
    parser.add_argument('--masks', required=True, help='Input segmentation masks (TIFF)')
    parser.add_argument('--prefix', required=True, help='Output prefix for files')
    parser.add_argument('--model_name', default='general_2d', help='Pretrained model name')
    parser.add_argument('--mode', default='greedy', choices=['greedy', 'greedy_nodiv', 'ilp'],
                       help='Tracking mode')
    parser.add_argument('--use_distance', action='store_true', default=True,
                       help='Use distance-based tracking')
    parser.add_argument('--no_distance', action='store_true', 
                       help='Disable distance-based tracking')
    parser.add_argument('--max_distance', type=int, default=30,
                       help='Maximum distance for linking cells')
    parser.add_argument('--allow_divisions', action='store_true', default=False,
                       help='Allow cell divisions')
    parser.add_argument('--save_graph', action='store_true', default=False,
                       help='Save tracking graph as pickle file')
    parser.add_argument('--sample_id', default='sample', help='Sample ID for summary')
    
    args = parser.parse_args()
    
    # Handle distance flag logic
    use_distance = args.use_distance and not args.no_distance
    
    print(f"Trackastra Cell Tracking")
    print(f"========================")
    print(f"Images: {args.images}")
    print(f"Masks: {args.masks}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {args.mode}")
    print(f"Use distance: {use_distance}")
    print(f"Max distance: {args.max_distance}")
    print(f"Allow divisions: {args.allow_divisions}")
    print(f"Save graph: {args.save_graph}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load images and masks
    print("\nLoading images and masks...")
    try:
        imgs = imread(args.images)
        # imgs = ski.transform.resize(imgs, (imgs.shape[0], imgs.shape[1] // 2, imgs.shape[2] // 2, imgs.shape[3] // 2), anti_aliasing=True)  # downsample the images by two to speed up the process

        masks = imread(args.masks)
        print(f"Images shape: {imgs.shape}")
        print(f"Masks shape: {masks.shape}")
        print(f"Images dtype: {imgs.dtype}")
        print(f"Masks dtype: {masks.dtype}")
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)
    
    # Ensure masks are uint16 and handle different dimensionalities
    if masks.dtype != np.uint16:
        print(f"Converting masks from {masks.dtype} to uint16")
        masks = masks.astype(np.uint16)
    
    # # Handle different image/mask shapes
    # if len(imgs.shape) != len(masks.shape):
    #     print(f"Image and mask dimensions mismatch: {imgs.shape} vs {masks.shape}")
    #     # Try to handle common cases
    #     if len(imgs.shape) == 4 and len(masks.shape) == 3:
    #         # Time series image with single channel masks
    #         print("Assuming time series with single channel masks")
    #     elif len(imgs.shape) == 3 and len(masks.shape) == 3:
    #         # Could be (T, H, W) for both or (C, H, W) vs (T, H, W)
    #         print("3D images and masks detected")
    
    # print(f"Final images shape: {imgs.shape}")
    # print(f"Final masks shape: {masks.shape}")
    
    
    # Handle different image/mask shapes
    print("Processing input data dimensions...")

    if len(imgs.shape) == 4 and len(masks.shape) == 3:
        # Images: (T, C, H, W), Masks: (T, H, W)
        print(f"Multi-channel time series detected")
        print(f"  Images: {imgs.shape} (Time, Channels, Height, Width)")
        print(f"  Masks:  {masks.shape} (Time, Height, Width)")
        
        # Check spatial dimensions
        target_h, target_w = imgs.shape[2], imgs.shape[3]
        current_h, current_w = masks.shape[1], masks.shape[2]
        
        # if (target_h == current_h) and (target_w == current_w):
        #     print(f"✓ Spatial dimensions match: {target_h}×{target_w}")
        # else:
        #     print(f"⚠ Spatial dimensions differ: images {target_h}×{target_w}, masks {current_h}×{current_w}")

        # here donwsize the images to match the masks if they differ
        if (target_h == current_h) and (target_w == current_w):
            print(f"✓ Spatial dimensions match: {target_h}×{target_w}")
        else:
            print(f"⚠ Spatial dimensions differ: images {target_h}×{target_w}, masks {current_h}×{current_w}")
            print(f"→ Downsizing images to match mask dimensions for efficiency...")
            
            # Resize images to match mask spatial dimensions (more efficient)
            imgs_resized = np.zeros((imgs.shape[0], imgs.shape[1], current_h, current_w), dtype=imgs.dtype)
            for t in range(imgs.shape[0]):
                for c in range(imgs.shape[1]):
                    imgs_resized[t, c] = ski.transform.resize(
                        imgs[t, c], 
                        (current_h, current_w), 
                        order=1,  # Linear interpolation for images
                        preserve_range=True,
                        anti_aliasing=True
                    ).astype(imgs.dtype)
            imgs = imgs_resized
            print(f"✓ Images resized to: {imgs.shape}")



        # Use first channel for tracking
        print(f"→ Using first channel from {imgs.shape[1]} channels for tracking")
        imgs_for_tracking = imgs[:, 0, :, :]  # Take first channel: (T, H, W)

        
    elif len(imgs.shape) == 3 and len(masks.shape) == 3:
        print(f"3D time series detected")
        print(f"  Images: {imgs.shape}")
        print(f"  Masks:  {masks.shape}")
        imgs_for_tracking = imgs
        
    else:
        print(f"Unexpected dimensions - Images: {imgs.shape}, Masks: {masks.shape}")
        imgs_for_tracking = imgs

    print(f"Final images shape for tracking: {imgs_for_tracking.shape}")
    print(f"Final masks shape: {masks.shape}")
        
    
    
    
    
    # Validate input data
    if np.max(masks) == 0:
        print("Warning: No segmented objects found in masks")
    else:
        print(f"Found {np.max(masks)} unique objects in masks")
    
    # Load pretrained model
    print(f"\nLoading Trackastra model: {args.model_name}")
    try:
        model = Trackastra.from_pretrained(args.model_name, device=device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Available models might include: 'general_2d', 'general_3d'")
        sys.exit(1)
    
    # Set up tracking parameters
    tracking_params = {
        'mode': args.mode,
        'use_distance': use_distance,
        'max_distance': args.max_distance,
        'allow_divisions': args.allow_divisions
    }
    print(f"\nTracking parameters: {tracking_params}")
    
    # Track the cells
    print("\nStarting cell tracking...")
    try:
        track_graph = model.track(
            #imgs, 
            imgs_for_tracking,  # Use processed images instead of imgs
            masks,
            mode=tracking_params['mode'],
            use_distance=tracking_params['use_distance'],
            max_distance=tracking_params['max_distance'],
            allow_divisions=tracking_params['allow_divisions']
        )
        print("Tracking completed successfully")
    except Exception as e:
        print(f"Error during tracking: {e}")
        print("This might be due to:")
        print("- Incompatible image/mask dimensions")
        print("- No objects to track in the masks")
        print("- Memory issues with large datasets")
        sys.exit(1)
    
    # Save tracking graph if requested
    if args.save_graph:
        try:
            graph_file = f"{args.prefix}_tracking_graph.pkl"
            with open(graph_file, "wb") as f:
                pickle.dump(track_graph, f)
            print(f"Tracking graph saved: {graph_file}")
        except Exception as e:
            print(f"Warning: Could not save tracking graph: {e}")
    
    # Write to cell tracking challenge format
    print("\nConverting to CTC format...")
    try:
        # Create output directory for CTC format
        ctc_output_dir = Path("./ctc_output")
        ctc_output_dir.mkdir(exist_ok=True)
        
        ctc_tracks, masks_tracked = graph_to_ctc(
            track_graph,
            masks,
            outdir=str(ctc_output_dir)
        )
        print("CTC conversion completed")
    except Exception as e:
        print(f"Error during CTC conversion: {e}")
        # Create fallback tracked masks
        masks_tracked = masks.copy()
        ctc_tracks = None
        print("Using original masks as fallback")
    
    # Save tracked masks
    try:
        tracked_masks_file = f"{args.prefix}_tracked_masks.tif"
        imwrite(tracked_masks_file, masks_tracked.astype(np.uint16))
        print(f"Tracked masks saved: {tracked_masks_file}")
    except Exception as e:
        print(f"Error saving tracked masks: {e}")
        sys.exit(1)
    
    # Save CTC tracks file
    try:
        ctc_tracks_file = f"{args.prefix}_ctc_tracks.txt"
        ctc_output_path = Path("./ctc_output")
        track_files = list(ctc_output_path.glob("res_track*.txt"))
        
        if track_files:
            # Use the first track file found
            import shutil
            shutil.copy(track_files[0], ctc_tracks_file)
            print(f"CTC tracks file saved: {ctc_tracks_file}")
        else:
            # Create a basic tracks file from the ctc_tracks data
            with open(ctc_tracks_file, "w") as f:
                if ctc_tracks is not None and len(ctc_tracks) > 0:
                    for track in ctc_tracks:
                        f.write(f"{track}\n")
                else:
                    f.write("# No tracks found or tracking failed\n")
            print(f"CTC tracks file created: {ctc_tracks_file}")
    except Exception as e:
        print(f"Error saving CTC tracks: {e}")
        # Create empty file to avoid pipeline failure
        ctc_tracks_file = f"{args.prefix}_ctc_tracks.txt"
        with open(ctc_tracks_file, "w") as f:
            f.write(f"# Error during track export: {e}\n")
        print(f"Empty CTC tracks file created: {ctc_tracks_file}")
    
    # Generate summary
    try:
        summary_file = f"{args.prefix}_tracking_summary.txt"
        unique_tracks = len(np.unique(masks_tracked)) - 1  # Subtract background
        
        summary_info = f"""Tracking Summary for {args.sample_id}
========================================
Model: {args.model_name}
Mode: {args.mode}
Use distance: {use_distance}
Max distance: {args.max_distance}
Allow divisions: {args.allow_divisions}
Device: {device}

Input Data:
- Images file: {args.images}
- Masks file: {args.masks}
- Images shape: {imgs.shape}
- Masks shape: {masks.shape}

Results:
- Input frames: {imgs.shape[0] if len(imgs.shape) > 2 else 1}
- Unique objects tracked: {unique_tracks}
- Original mask max ID: {np.max(masks)}
- Tracked mask max ID: {np.max(masks_tracked)}

Output Files:
- {args.prefix}_tracked_masks.tif
- {args.prefix}_ctc_tracks.txt
- {args.prefix}_tracking_summary.txt
{f"- {args.prefix}_tracking_graph.pkl" if args.save_graph else ""}

Tracking Status: {'SUCCESS' if ctc_tracks is not None else 'PARTIAL (fallback used)'}
"""
        
        with open(summary_file, "w") as f:
            f.write(summary_info)
        print(f"Summary saved: {summary_file}")
        
    except Exception as e:
        print(f"Error creating summary: {e}")
        summary_file = f"{args.prefix}_tracking_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Error generating summary: {e}\n")
    
    print("\n" + "="*50)
    print("Tracking pipeline completed successfully!")
    print(f"Check the summary file for details: {summary_file}")
    print("="*50)


if __name__ == "__main__":
    main()