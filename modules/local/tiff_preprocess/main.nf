process TIFF_PREPROCESS {
    tag "$meta.id"
    label 'process_medium'

    conda "conda-forge::python=3.9 conda-forge::tifffile conda-forge::scikit-image conda-forge::numpy conda-forge::tqdm"

    input:
    tuple val(meta), path(tiff)

    output:
    //tuple val(meta), path("*_actin_resized.tif"), emit: actin_channel
    //tuple val(meta), path("*_calcium_resized.tif"), emit: calcium_channel, optional: true
    //tuple val(meta), path("*_caspase_resized.tif"), emit: caspase_channel, optional: true
    tuple val(meta), path("*_caspase_resized.tif"), emit: caspase_channel
    tuple val(meta), path("*_actin_resized.tif"), emit: actin_channel, optional: true
    tuple val(meta), path("*_calcium_resized.tif"), emit: calcium_channel, optional: true
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    #!/usr/bin/env python3

    from tifffile import imread, imwrite
    from skimage.transform import resize
    import numpy as np
    from tqdm import tqdm

    # Read the multi-channel TIFF
    image = imread("${tiff}")
    
    # Extract channels (assuming 4 channels: Caspase3, Actin/Tubulin, Calcium, Brightfield)
    # Channel 0: Caspase 3
    # Channel 1: Actin/Tubulin (this is what we'll use for segmentation)
    # Channel 2: Calcium
    # Channel 3: Brightfield
    
    if len(image.shape) == 4:  # Time, Channel, Height, Width
        image_caspase_channel = image[:, 0, ...]
        image_actin_channel = image[:, 1, ...]
        image_calcium_channel = image[:, 2, ...]
        image_brightfield_channel = image[:, 3, ...]
    elif len(image.shape) == 3:  # Channel, Height, Width (single timepoint)
        image_caspase_channel = image[0, ...]
        image_actin_channel = image[1, ...]
        image_calcium_channel = image[2, ...]
        image_brightfield_channel = image[3, ...]
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    
    # Resize images by half for performance
    if len(image_actin_channel.shape) == 3:  # Time series
        image_actin_resized = resize(
            image_actin_channel, 
            (image_actin_channel.shape[0], image_actin_channel.shape[1] // 2, image_actin_channel.shape[2] // 2), 
            anti_aliasing=True
        )
        image_calcium_resized = resize(
            image_calcium_channel, 
            (image_calcium_channel.shape[0], image_calcium_channel.shape[1] // 2, image_calcium_channel.shape[2] // 2), 
            anti_aliasing=True
        )
        image_caspase_resized = resize(
            image_caspase_channel, 
            (image_caspase_channel.shape[0], image_caspase_channel.shape[1] // 2, image_caspase_channel.shape[2] // 2), 
            anti_aliasing=True
        )
    else:  # Single frame
        image_actin_resized = resize(
            image_actin_channel, 
            (image_actin_channel.shape[0] // 2, image_actin_channel.shape[1] // 2), 
            anti_aliasing=True
        )
        image_calcium_resized = resize(
            image_calcium_channel, 
            (image_calcium_channel.shape[0] // 2, image_calcium_channel.shape[1] // 2), 
            anti_aliasing=True
        )
        image_caspase_resized = resize(
            image_caspase_channel, 
            (image_caspase_channel.shape[0] // 2, image_caspase_channel.shape[1] // 2), 
            anti_aliasing=True
        )
    
    # Save the resized channels
    imwrite("${prefix}_actin_resized.tif", image_actin_resized.astype(np.float32))
    imwrite("${prefix}_calcium_resized.tif", image_calcium_resized.astype(np.float32))
    imwrite("${prefix}_caspase_resized.tif", image_caspase_resized.astype(np.float32))
    
    print(f"Processed ${tiff}")
    print(f"Original shape: {image.shape}")
    print(f"Actin channel resized shape: {image_actin_resized.shape}")

    # Create versions file
    import sys
    import tifffile
    import skimage
    
    with open("versions.yml", "w") as f:
        f.write('"${task.process}":\\n')
        f.write(f'    python: "{sys.version.split()[0]}"\\n')
        f.write(f'    tifffile: "{tifffile.__version__}"\\n')
        f.write(f'    scikit-image: "{skimage.__version__}"\\n')
        f.write(f'    numpy: "{np.__version__}"\\n')
    """
}
