/content/nextflow run main.nf \
  --input samplesheet_colab.csv \
  --channels_to_display "0,1,2" \
  --analysis_channels "actin_tubulin:1,csfe:0" \
  --outdir ./results \
  -profile conda \
  -resume
