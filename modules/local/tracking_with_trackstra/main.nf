process TRACKING_WITH_TRACKSTRA {
    tag "$meta.id"
    label 'process_high'

    conda "${moduleDir}/environment.yml"

    input:
    tuple val(meta), path(tiff), path(tiff_mask)

    output:
    tuple val(meta), path("${meta.id}_tracked_masks.tif")      , emit: tracked_masks
    tuple val(meta), path("${meta.id}_ctc_tracks.txt")         , emit: ctc_tracks  
    tuple val(meta), path("${meta.id}_tracking_graph.pkl")     , emit: track_graph, optional: true
    tuple val(meta), path("${meta.id}_tracking_summary.txt")   , emit: summary
    path "versions.yml"                                        , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def mode = task.ext.mode ?: 'greedy'
    def use_distance = task.ext.use_distance ?: true
    def max_distance = task.ext.max_distance ?: 30
    def allow_divisions = task.ext.allow_divisions ?: false
    def model_name = task.ext.model_name ?: 'general_2d'
    def save_graph = task.ext.save_graph ?: false
    
    // Build command line arguments
    def distance_flag = use_distance ? '--use_distance' : '--no_distance'
    def divisions_flag = allow_divisions ? '--allow_divisions' : ''
    def graph_flag = save_graph ? '--save_graph' : ''
    
    """
    # Fix matplotlib backend for headless environment
    export MPLBACKEND=Agg
    
    python ${projectDir}/bin/tracking_with_trackstra.py \\
        --images ${tiff} \\
        --masks ${tiff_mask} \\
        --prefix ${prefix} \\
        --model_name ${model_name} \\
        --mode ${mode} \\
        ${distance_flag} \\
        --max_distance ${max_distance} \\
        ${divisions_flag} \\
        ${graph_flag} \\
        --sample_id ${meta.id} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //g')
        torch: \$(python -c "import torch; print(torch.__version__)")
        trackastra: \$(python -c "import trackastra; print(trackastra.__version__)" 2>/dev/null || echo "unknown")
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)")
        numpy: \$(python -c "import numpy; print(numpy.__version__)")
        scikit-image: \$(python -c "import skimage; print(skimage.__version__)")
    END_VERSIONS
    """

    stub:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    # Fix matplotlib backend for headless environment
    export MPLBACKEND=Agg
    
    touch ${prefix}_tracked_masks.tif
    touch ${prefix}_ctc_tracks.txt
    touch ${prefix}_tracking_summary.txt
    echo "stub run" > ${prefix}_tracking_summary.txt

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //g')
        torch: "stub"
        trackastra: "stub" 
        tifffile: "stub"
        numpy: "stub"
        scikit-image: "stub"
    END_VERSIONS
    """
}