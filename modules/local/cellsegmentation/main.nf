process CELLSEGMENTATION {
    tag "$meta.id"
    label 'process_high'
    label 'process_low'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/python:3.9--1' :
        'biocontainers/python:3.9--1' }"

    input:
    tuple val(meta), path(tiff), path(tiff_mask)

    output:
    tuple val(meta), path("predictions.tiff")           , emit: predictions
    tuple val(meta), path("overlay_predictions.tiff")  , emit: overlays
    tuple val(meta), path("masked_predictions.tiff")   , emit: masked_predictions
    path "versions.yml"                                 , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def interval = task.ext.interval ?: 10
    def green_masking_thr = task.ext.green_masking_thr ?: 140
    def smallest_area_th = task.ext.smallest_area_th ?: 150
    """
    cellsegmentation.py \\
        --tiff ${tiff} \\
        --tiff_mask ${tiff_mask} \\
        --interval ${interval} \\
        --green_masking_thr ${green_masking_thr} \\
        --smallest_area_th ${smallest_area_th} \\
        --prefix ${prefix} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //g')
        numpy: \$(python -c "import numpy; print(numpy.__version__)")
        scikit-image: \$(python -c "import skimage; print(skimage.__version__)")
        scikit-learn: \$(python -c "import sklearn; print(sklearn.__version__)")
        opencv-python: \$(python -c "import cv2; print(cv2.__version__)")
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)")
    END_VERSIONS
    """

    stub:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch predictions.tiff
    touch overlay_predictions.tiff
    touch masked_predictions.tiff

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //g')
        numpy: \$(python -c "import numpy; print(numpy.__version__)")
        scikit-image: \$(python -c "import skimage; print(skimage.__version__)")
        scikit-learn: \$(python -c "import sklearn; print(sklearn.__version__)")
        opencv-python: \$(python -c "import cv2; print(cv2.__version__)")
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)")
    END_VERSIONS
    """
}
