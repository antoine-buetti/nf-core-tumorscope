process CELLSEGMENTATION {
    tag "$meta.id"
    label 'process_high'

    //conda "conda-forge::python=3.9 conda-forge::numpy conda-forge::opencv conda-forge::scikit-image conda-forge::imageio conda-forge::scipy conda-forge::scikit-learn conda-forge::tifffile conda-forge::tqdm"
    conda "${moduleDir}/environment.yml"

    input:
    tuple val(meta), path(tiff), path(tiff_mask)

    output:
    tuple val(meta), path("${meta.id}_predictions.tiff")           , emit: predictions
    tuple val(meta), path("${meta.id}_overlay_predictions.tiff")  , emit: overlays
    tuple val(meta), path("${meta.id}_masked_predictions.tiff")   , emit: masked_predictions
    path "versions.yml"                                           , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def interval = task.ext.interval ?: 10
    def green_masking_thr = task.ext.green_masking_thr ?: 140
    def smallest_area_th = task.ext.smallest_area_th ?: 150
    """
    # Works on seqera platform but not locally:
    #chmod +x /usr/local/bin/cellsegmentation.py
    #/usr/local/bin/cellsegmentation.py \\

    python ${projectDir}/bin/cellsegmentation.py \\
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
    touch ${prefix}_predictions.tiff
    touch ${prefix}_overlay_predictions.tiff
    touch ${prefix}_masked_predictions.tiff

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
