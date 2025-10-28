<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/nf-core-tumorscope_logo_dark.png">
    <img alt="nf-core/tumorscope" src="docs/images/nf-core-tumorscope_logo_light.png">
  </picture>
</h1>

[![GitHub Actions CI Status](https://github.com/nf-core/tumorscope/actions/workflows/ci.yml/badge.svg)](https://github.com/nf-core/tumorscope/actions/workflows/ci.yml)
[![GitHub Actions Linting Status](https://github.com/nf-core/tumorscope/actions/workflows/linting.yml/badge.svg)](https://github.com/nf-core/tumorscope/actions/workflows/linting.yml)[![AWS CI](https://img.shields.io/badge/CI%20tests-full%20size-FF9900?labelColor=000000&logo=Amazon%20AWS)](https://nf-co.re/tumorscope/results)[![Cite with Zenodo](http://img.shields.io/badge/DOI-10.5281/zenodo.XXXXXXX-1073c8?labelColor=000000)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![nf-test](https://img.shields.io/badge/unit_tests-nf--test-337ab7.svg)](https://www.nf-test.com)

[![Nextflow](https://img.shields.io/badge/nextflow%20DSL2-%E2%89%A524.04.2-23aa62.svg)](https://www.nextflow.io/)
[![run with conda](http://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/en/latest/)
[![run with docker](https://img.shields.io/badge/run%20with-docker-0db7ed?labelColor=000000&logo=docker)](https://www.docker.com/)
[![run with singularity](https://img.shields.io/badge/run%20with-singularity-1d355c.svg?labelColor=000000)](https://sylabs.io/docs/)
[![Launch on Seqera Platform](https://img.shields.io/badge/Launch%20%F0%9F%9A%80-Seqera%20Platform-%234256e7)](https://cloud.seqera.io/launch?pipeline=https://github.com/nf-core/tumorscope)

[![Get help on Slack](http://img.shields.io/badge/slack-nf--core%20%23tumorscope-4A154B?labelColor=000000&logo=slack)](https://nfcore.slack.com/channels/tumorscope)[![Follow on Twitter](http://img.shields.io/badge/twitter-%40nf__core-1DA1F2?labelColor=000000&logo=twitter)](https://twitter.com/nf_core)[![Follow on Mastodon](https://img.shields.io/badge/mastodon-nf__core-6364ff?labelColor=FFFFFF&logo=mastodon)](https://mstdn.science/@nf_core)[![Watch on YouTube](http://img.shields.io/badge/youtube-nf--core-FF0000?labelColor=000000&logo=youtube)](https://www.youtube.com/c/nf-core)

## Introduction

**nf-core/tumorscope** is a bioinformatics pipeline for analyzing multi-channel microscopy videos in TIFF format. The pipeline processes time-lapse microscopy data with multiple fluorescent channels (Caspase 3, Actin/Tubulin, Calcium, Brightfield) and performs cell segmentation using Cellpose on the Actin/Tubulin channel.

The pipeline performs the following steps:

1. **Multi-channel TIFF preprocessing** - Extracts individual channels from multi-channel TIFF files and resizes images for optimal processing
2. **Cell segmentation** - Uses Cellpose with the cyto3 model to segment cells from the Actin/Tubulin channel
3. **Quality control reporting** - Generates comprehensive reports with MultiQC

## Usage

> [!NOTE]
> If you are new to Nextflow and nf-core, please refer to [this page](https://nf-co.re/docs/usage/installation) on how to set-up Nextflow. Make sure to [test your setup](https://nf-co.re/docs/usage/introduction#how-to-run-a-pipeline) with `-profile test` before running the workflow on actual data.

First, prepare a samplesheet with your input data that looks as follows:

`samplesheet.csv`:

```csv
sample,tiff
sample1,/path/to/your/microscopy_video1.tif
sample2,/path/to/your/microscopy_video2.tif
```

Each row represents a multi-channel TIFF file containing microscopy time-lapse data. The TIFF files should have 4 channels in the following order:
- Channel 0: Caspase 3
- Channel 1: Actin/Tubulin (used for segmentation)
- Channel 2: Calcium
- Channel 3: Brightfield

Now, you can run the pipeline using:

<!-- TODO nf-core: update the following command to include all required parameters for a minimal example -->

```bash
nextflow run nf-core/tumorscope \
   -profile <docker/singularity/.../institute> \
   --input samplesheet.csv \
   --outdir results \
   --diameter 45 \
   --model_type cyto3
```

You can customize the Cellpose segmentation parameters:
- `--diameter 45`: Expected cell diameter in pixels (number, default: 45)
- `--model_type cyto3`: Cellpose model to use (string, options: cyto, cyto2, cyto3, nuclei)
- `--flow_threshold 0.8`: Flow error threshold (number, default: 0.8)
- `--cellprob_threshold -1.0`: Cell probability threshold (number, default: -1.0)
- `--gpu false`: Enable GPU acceleration (boolean, default: false)

> [!WARNING]
> Please provide pipeline parameters via the CLI or Nextflow `-params-file` option. Custom config files including those provided by the `-c` Nextflow option can be used to provide any configuration _**except for parameters**_; see [docs](https://nf-co.re/docs/usage/getting_started/configuration#custom-configuration-files).

For more details and further functionality, please refer to the [usage documentation](https://nf-co.re/tumorscope/usage) and the [parameter documentation](https://nf-co.re/tumorscope/parameters).

## Pipeline output

To see the results of an example test run with a full size dataset refer to the [results](https://nf-co.re/tumorscope/results) tab on the nf-core website pipeline page.
For more details about the output files and reports, please refer to the
[output documentation](https://nf-co.re/tumorscope/output).

## Credits

nf-core/tumorscope was originally written by Seqera AI.

We thank the following people for their extensive assistance in the development of this pipeline:

<!-- TODO nf-core: If applicable, make list of people who have also contributed -->

## Contributions and Support

If you would like to contribute to this pipeline, please see the [contributing guidelines](.github/CONTRIBUTING.md).

For further information or help, don't hesitate to get in touch on the [Slack `#tumorscope` channel](https://nfcore.slack.com/channels/tumorscope) (you can join with [this invite](https://nf-co.re/join/slack)).

## Citations

<!-- TODO nf-core: Add citation for pipeline after first release. Uncomment lines below and update Zenodo doi and badge at the top of this file. -->
<!-- If you use nf-core/tumorscope for your analysis, please cite it using the following doi: [10.5281/zenodo.XXXXXX](https://doi.org/10.5281/zenodo.XXXXXX) -->

<!-- TODO nf-core: Add bibliography of tools and data used in your pipeline -->

An extensive list of references for the tools used by the pipeline can be found in the [`CITATIONS.md`](CITATIONS.md) file.

You can cite the `nf-core` publication as follows:

> **The nf-core framework for community-curated bioinformatics pipelines.**
>
> Philip Ewels, Alexander Peltzer, Sven Fillinger, Harshil Patel, Johannes Alneberg, Andreas Wilm, Maxime Ulysse Garcia, Paolo Di Tommaso & Sven Nahnsen.
>
> _Nat Biotechnol._ 2020 Feb 13. doi: [10.1038/s41587-020-0439-x](https://dx.doi.org/10.1038/s41587-020-0439-x).
