process CELLPOSE_SEGMENT {
    tag "$meta.id"
    label 'process_high'

    conda "conda-forge::python=3.9 conda-forge::cellpose conda-forge::tifffile conda-forge::numpy conda-forge::tqdm"

    input:
    tuple val(meta), path(image)

    output:
    tuple val(meta), path("*_cell_masks.tif"), emit: masks
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def diameter = task.ext.diameter ?: '45'
    def flow_threshold = task.ext.flow_threshold ?: '0.8'
    def cellprob_threshold = task.ext.cellprob_threshold ?: '-1.0'
    def model_type = task.ext.model_type ?: 'cyto3'
    def gpu = task.ext.gpu ?: false
    def gpu_python = gpu ? 'True' : 'False'
    """
    #!/usr/bin/env python3

    from cellpose import models
    from tifffile import imread, imwrite
    import numpy as np
    from tqdm import tqdm

    # Load the preprocessed actin channel image
    image = imread("${image}")
    
    # Initialize Cellpose model
    cellpose_model = models.CellposeModel(gpu=${gpu_python}, model_type='${model_type}')
    
    # Initialize masks array
    masks = np.zeros_like(image, dtype=np.uint16)
    
    # Process each frame if it's a time series, otherwise process single frame
    if len(image.shape) == 3:  # Time series
        print(f"Processing {image.shape[0]} frames...")
        for i in tqdm(range(image.shape[0])):
            masks[i], flows, styles = cellpose_model.eval(
                image[i], 
                diameter=${diameter}, 
                do_3D=False, 
                channels=[0, 0], 
                normalize=True, 
                flow_threshold=${flow_threshold}, 
                cellprob_threshold=${cellprob_threshold}
            )
    else:  # Single frame
        print("Processing single frame...")
        masks, flows, styles = cellpose_model.eval(
            image, 
            diameter=${diameter}, 
            do_3D=False, 
            channels=[0, 0], 
            normalize=True, 
            flow_threshold=${flow_threshold}, 
            cellprob_threshold=${cellprob_threshold}
        )
    
    # Save masks
    imwrite("${prefix}_cell_masks.tif", masks.astype(np.uint16))
    
    print(f"Segmentation completed for ${image}")
    print(f"Masks shape: {masks.shape}")
    print(f"Number of unique cells found: {len(np.unique(masks)) - 1}")  # -1 to exclude background

    # Create versions file
    import sys
    import cellpose
    
    with open("versions.yml", "w") as f:
        f.write('"${task.process}":\\n')
        f.write(f'    python: "{sys.version.split()[0]}"\\n')
        f.write(f'    numpy: "{np.__version__}"\\n')
    """
}
