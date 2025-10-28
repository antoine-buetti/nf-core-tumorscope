/content/nextflow run main.nf \
  --input samplesheet_colab.csv \
  --outdir ./results \
  --diameter 45 \
  --model_type cyto3 \
  --gpu true \
  --interval 10 \
  --green_masking_thr 140 \
  --spot_sigma 6 \
  --outline_sigma 2 \
  --frame_index 1 \
  -profile conda \
  -resume

