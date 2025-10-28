nextflow run main.nf \
  --input samplesheet_local.csv \
  --outdir ./results \
  --diameter 45 \
  --model_type cyto3 \
  --gpu false \
  --interval 10 \
  --green_masking_thr 140 \
  -profile conda \
  -resume

