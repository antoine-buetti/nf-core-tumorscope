/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT MODULES / SUBWORKFLOWS / FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
//include { CELLSEGMENTATION       } from '../modules/local/cellsegmentation'
include { CELLSEGMENTATION       } from '../modules/local/cellsegmentation_GPU' // accelerated
include { TIFF_PREPROCESS        } from '../modules/local/tiff_preprocess/main'
include { INSTANCE_LABELS        } from '../modules/local/instance_labels/main'     
include { CELLPOSE_SEGMENT       } from '../modules/local/cellpose_segment/main'
include { TRACKING_WITH_TRACKSTRA } from '../modules/local/tracking_with_trackstra/main'
include { TRACKING_VISUALIZATION } from '../modules/local/tracking_visualization/main'
include { MULTIQC                } from '../modules/nf-core/multiqc/main'
include { paramsSummaryMap       } from 'plugin/nf-schema'
include { paramsSummaryMultiqc   } from '../subworkflows/nf-core/utils_nfcore_pipeline'
include { softwareVersionsToYAML } from '../subworkflows/nf-core/utils_nfcore_pipeline'
include { methodsDescriptionText } from '../subworkflows/local/utils_nfcore_tumorscope_pipeline'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow tumorscope {

    take:
    ch_samplesheet // channel: samplesheet read in from --input
    main:

    ch_versions = Channel.empty()
    ch_multiqc_files = Channel.empty()
    
    //
    // MODULE: Preprocess multi-channel TIFF files
    //
    TIFF_PREPROCESS (
        ch_samplesheet
    )
    ch_versions = ch_versions.mix(TIFF_PREPROCESS.out.versions.first())
    
    //
    // MODULE: Run Cellpose segmentation on actin channel
    //
    CELLPOSE_SEGMENT (
        TIFF_PREPROCESS.out.caspase_channel
    )
    ch_versions = ch_versions.mix(CELLPOSE_SEGMENT.out.versions.first())

    ch_combo = ch_samplesheet.join(CELLPOSE_SEGMENT.out.masks) // tuple val(meta), path(tiff), path(tiff_mask)
    // ch_combo.view { "Combo: $it" }

    //
    // MODULE: Run Cell Segmentation
    //
    CELLSEGMENTATION (
        ch_combo
    )
    ch_versions = ch_versions.mix(CELLSEGMENTATION.out.versions.first())

    //
    // MODULE: Run Cell Segmentation
    //
    INSTANCE_LABELS (
        ch_combo
    )
    ch_versions = ch_versions.mix(INSTANCE_LABELS.out.versions.first())


    //
    // MODULE: Run Tracking with Trackastra
    //
    TRACKING_WITH_TRACKSTRA(
        ch_combo
    )
    ch_versions = ch_versions.mix(TRACKING_WITH_TRACKSTRA.out.versions.first())

    // // Create input for visualization: combine original images with tracked masks
    // ch_visualization_input = ch_samplesheet
    //     .join(TRACKING_WITH_TRACKSTRA.out.tracked_masks, by: [0])
    //     .map { meta, tiff, tracked_masks -> 
    //         [meta, tiff, tracked_masks] 
    //     }

    //
    // MODULE: Run visualization with tracked data
    //
    TRACKING_VISUALIZATION(
        // ch_visualization_input
        ch_combo
    )
    ch_versions = ch_versions.mix(TRACKING_VISUALIZATION.out.versions.first())
    
    // // Access outputs
    // animations = TRACKING_VISUALIZATION.out.animation         // HTML animations
    // trajectories = TRACKING_VISUALIZATION.out.trajectories   // JSON trajectory data
    // intensities = TRACKING_VISUALIZATION.out.intensities     // JSON intensity data
    // summaries = TRACKING_VISUALIZATION.out.summary           // Analysis summaries
    
    // // Optional: Process results further
    // animations.view { meta, file -> "Animation for ${meta.id}: ${file}" }
    // summaries.view { meta, file -> "Summary for ${meta.id}: ${file}" }


    //
    // Collate and save software versions
    //
    softwareVersionsToYAML(ch_versions)
        .collectFile(
            storeDir: "${params.outdir}/pipeline_info",
            name: 'nf_core_'  +  'tumorscope_software_'  + 'mqc_'  + 'versions.yml',
            sort: true,
            newLine: true
        ).set { ch_collated_versions }


    //
    // MODULE: MultiQC
    //
    ch_multiqc_config        = Channel.fromPath(
        "$projectDir/assets/multiqc_config.yml", checkIfExists: true)
    ch_multiqc_custom_config = params.multiqc_config ?
        Channel.fromPath(params.multiqc_config, checkIfExists: true) :
        Channel.empty()
    ch_multiqc_logo          = params.multiqc_logo ?
        Channel.fromPath(params.multiqc_logo, checkIfExists: true) :
        Channel.empty()

    summary_params      = paramsSummaryMap(
        workflow, parameters_schema: "nextflow_schema.json")
    ch_workflow_summary = Channel.value(paramsSummaryMultiqc(summary_params))
    ch_multiqc_files = ch_multiqc_files.mix(
        ch_workflow_summary.collectFile(name: 'workflow_summary_mqc.yaml'))
    ch_multiqc_custom_methods_description = params.multiqc_methods_description ?
        file(params.multiqc_methods_description, checkIfExists: true) :
        file("$projectDir/assets/methods_description_template.yml", checkIfExists: true)
    ch_methods_description                = Channel.value(
        methodsDescriptionText(ch_multiqc_custom_methods_description))

    ch_multiqc_files = ch_multiqc_files.mix(ch_collated_versions)
    ch_multiqc_files = ch_multiqc_files.mix(
        ch_methods_description.collectFile(
            name: 'methods_description_mqc.yaml',
            sort: true
        )
    )

    
    MULTIQC (
        ch_multiqc_files.collect(),
        ch_multiqc_config.toList(),
        ch_multiqc_custom_config.toList(),
        ch_multiqc_logo.toList(),
        [],
        []
    )


    emit:multiqc_report = MULTIQC.out.report.toList() // channel: /path/to/multiqc_report.html
    versions       = ch_versions                 // channel: [ path(versions.yml) ]
/* 
 */    

}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
