process INSTANCE_LABELS {
    tag "$meta.id"
    label 'process_high'

    conda "${moduleDir}/environment.yml"

    input:
    tuple val(meta), path(tiff), path(tiff_mask)

    output:
    tuple val(meta), path("${meta.id}_instance_labels.tiff")     , emit: instance_labels
    tuple val(meta), path("${meta.id}_segmented_mask.tiff")     , emit: segmented_mask
    tuple val(meta), path("${meta.id}_masked_channels.tiff")    , emit: masked_channels
    tuple val(meta), path("${meta.id}_segmentation_overlay.png"), emit: overlay_plot
    path "versions.yml"                                         , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def spot_sigma = task.ext.spot_sigma ?: 1
    def outline_sigma = task.ext.outline_sigma ?: 1
    def frame_index = task.ext.frame_index ?: 1  // Default frame for time series
    """
    python ${projectDir}/bin/instance_labels.py \\
        --image ${tiff} \\
        --mask ${tiff_mask} \\
        --prefix ${prefix} \\
        --spot_sigma ${spot_sigma} \\
        --outline_sigma ${outline_sigma} \\
        --frame_index ${frame_index} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //g')
        numpy: \$(python -c "import numpy; print(numpy.__version__)")
        scikit-image: \$(python -c "import skimage; print(skimage.__version__)")
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)")
        pyclesperanto-prototype: \$(python -c "import pyclesperanto_prototype; print(pyclesperanto_prototype.__version__)")
        matplotlib: \$(python -c "import matplotlib; print(matplotlib.__version__)")
    END_VERSIONS
    """

    stub:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}_instance_labels.tiff
    touch ${prefix}_segmented_mask.tiff
    touch ${prefix}_masked_channels.tiff
    touch ${prefix}_segmentation_overlay.png

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //g')
        numpy: \$(python -c "import numpy; print(numpy.__version__)")
        scikit-image: \$(python -c "import skimage; print(skimage.__version__)")
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)")
        pyclesperanto-prototype: \$(python -c "import pyclesperanto_prototype; print(pyclesperanto_prototype.__version__)")
        matplotlib: \$(python -c "import matplotlib; print(matplotlib.__version__)")
    END_VERSIONS
    """
}
