#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import skimage
import imageio.v2 as imageio
from scipy import ndimage
import functools
from sklearn.ensemble import RandomForestClassifier
import tifffile
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_ubyte
from tqdm import tqdm


def normalize_grayscale(image, low_perc=2, high_perc=98):
    """Normalize grayscale image using percentile clipping."""
    low = np.percentile(image, low_perc)
    high = np.percentile(image, high_perc)
    clipped = np.clip(image, low, high)
    normalized = (clipped - low) / (high - low)
    return (normalized * 255).astype(np.uint8)


def remove_small_el(mask, area_thr=50, connectivity=8):
    """Remove small connected components from binary mask."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    
    mask_clean = np.zeros_like(mask)
    
    for label in range(1, num_labels):  # background label = 0
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= area_thr:
            mask_clean[labels == label] = 1  # Keep the component
    
    return mask_clean


def main():
    parser = argparse.ArgumentParser(description='Cell segmentation using Random Forest')
    parser.add_argument('--tiff', required=True, help='Input video TIFF file')
    parser.add_argument('--tiff_mask', required=True, help='Input video mask TIFF file')
    parser.add_argument('--interval', type=int, default=10, help='Frame interval for training')
    parser.add_argument('--green_masking_thr', type=int, default=140, help='Green channel masking threshold')
    parser.add_argument('--smallest_area_th', type=int, default=150, help='Smallest area threshold')
    parser.add_argument('--prefix', default='sample', help='Output file prefix')  # ADD THIS
    parser.add_argument('--n_estimators', type=int, default=300, help='Number of estimators for Random Forest')
    parser.add_argument('--max_depth', type=int, default=30, help='Maximum depth for Random Forest')
    parser.add_argument('--max_samples', type=float, default=0.5, help='Maximum samples for Random Forest')

    args = parser.parse_args()
    
    # Read input files
    print(f"Reading video from: {args.tiff}")
    image = imageio.imread(args.tiff)
    print(f"Video shape: {image.shape}")
    
    print(f"Reading cancer mask from: {args.tiff_mask}")
    cancer_masks = imageio.imread(args.tiff_mask)
    print(f"Cancer mask shape: {cancer_masks.shape}")
    
    nb_frames = image.shape[0]
    print(f"Number of frames: {nb_frames}")
    
    # Parameters
    red_channel = 1
    green_channel = 2
    
    # Feature extraction function
    feature_function = functools.partial(
        skimage.feature.multiscale_basic_features,
        intensity=True,
        edges=True,
        texture=True,
        sigma_min=1,
        sigma_max=6,
    )
    
    # Training data preparation
    print("Preparing training data...")
    green_masks = []
    green_frames = []
    training_red_frames = []
    
    # Storage for all features and labels
    X_train = []
    y_train = []
    
    i = 0
    while i < nb_frames:
        red_frame = image[i, red_channel, :, :]
        green_frame = image[i, green_channel, :, :]
        
        green_frames.append(green_frame)
        
        normalized_red = normalize_grayscale(red_frame)
        training_red_frames.append(normalized_red)
        
        green_mask = np.zeros_like(green_frame, dtype=np.uint8)
        green_mask[green_frame > args.green_masking_thr] = 1
        green_mask = remove_small_el(green_mask, area_thr=args.smallest_area_th)
        green_mask = (green_mask > 0).astype(np.uint8)
        green_masks.append(green_mask)
        
        # Extract features from red image
        features = feature_function(normalized_red)
        
        # Flatten features and labels
        h, w = green_mask.shape
        features_flat = features.reshape(-1, features.shape[-1])
        labels_flat = green_mask.flatten()
        
        # Use only labeled pixels (label 0 and 1)
        mask = (labels_flat == 0) | (labels_flat == 1)
        X_train.append(features_flat[mask])
        y_train.append(labels_flat[mask])
        
        i += args.interval
    
    # Stack all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Train classifier
    print("Training Random Forest classifier...")
    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        n_jobs=-1,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        class_weight="balanced",
    )
    classifier.fit(X_train, y_train)
    
    # Predict on all frames
    print("Predicting on all frames...")
    preds = []
    for i in tqdm(range(nb_frames)):
        red_frame = image[i, red_channel, :, :]
        test_img = normalize_grayscale(red_frame)
        
        features = feature_function(test_img)
        h, w = test_img.shape
        features_flat = features.reshape(-1, features.shape[-1])
        pred_flat = classifier.predict(features_flat)
        result = pred_flat.reshape(h, w)
        
        result_uint8 = (result * 255).astype(np.uint8)
        preds.append(result_uint8)
    
    # Save predictions
    print("Saving predictions...")
    tifffile.imwrite(f"{args.prefix}_predictions.tiff", np.stack(preds, axis=0))

    # Create overlays
    print("Creating overlays...")
    overlays = []
    for i in tqdm(range(nb_frames)):
        red_frame = normalize_grayscale(image[i, red_channel, :, :])
        overlay_rgb = mark_boundaries(img_as_ubyte(red_frame), preds[i], color=(1, 0, 0), mode='thick')
        overlays.append((overlay_rgb * 255).astype(np.uint8))
    
    tifffile.imwrite(f"{args.prefix}_overlay_predictions.tiff", np.stack(overlays, axis=0))

    # Post-processing: Remove overlaps with cancer masks
    print("Post-processing: removing cancer overlaps...")
    tcell_masks = np.array(preds)
    
    # Resize cancer masks if needed
    if cancer_masks.shape != tcell_masks.shape:
        cancer_masks = skimage.transform.resize(
            cancer_masks,
            tcell_masks.shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(cancer_masks.dtype)
    
    # Binarize masks
    for i in range(nb_frames):
        cancer_masks[i][cancer_masks[i] != 0] = 1
        tcell_masks[i][tcell_masks[i] != 0] = 1
    
    # Remove cancer regions from t-cell predictions
    final_pred = tcell_masks - cancer_masks
    for i in range(nb_frames):
        final_pred[i][final_pred[i] <= 0] = 0
    
    tifffile.imwrite(f"{args.prefix}_masked_predictions.tiff", np.stack(final_pred, axis=0).astype(np.uint8))
    print("Cell segmentation completed successfully!")


if __name__ == "__main__":
    main()