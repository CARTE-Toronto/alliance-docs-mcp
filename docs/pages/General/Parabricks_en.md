---
title: "Parabricks/en"
url: "https://docs.alliancecan.ca/wiki/Parabricks/en"
category: "General"
last_modified: "2020-09-09T12:06:02Z"
page_id: 14606
display_title: "Parabricks"
---

`<languages />`{=html}

Parabricks is a software suite for performing secondary analysis of next generation sequencing (NGS) DNA data. Parabricks is fast: its documentation claims that, thanks to its tight integration with GPUs, it is able to analyse 30x Whole Human Genome Sequencing (WGS) in hours as opposed to days with other techniques.

You can learn more at [www.nvidia.com/parabricks](http://www.nvidia.com/parabricks)

# Usage in Compute Canada Clusters {#usage_in_compute_canada_clusters}

**This software was provided freely by NVidia to help with research on COVID19 until Sunday, 17 May 2020.** Since this free period has expired, you must have your own license arrangement with NVidia in order to use Parabricks on Compute Canada equipment.

## Finding and loading Parabricks {#finding_and_loading_parabricks}

Parabricks can be looked for as a regular module through module spider:

Likewise, it can be loaded through LMOD modules:

## Example of use {#example_of_use}

Before you use Parabricks, make sure you have gone through the [Parabricks documentation](https://www.nvidia.com/en-us/docs/parabricks/), including their standalone tools and pipelines. Also make sure you know [how to request GPUs in Compute Canada clusters](https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm). Once you understand the above, you can submit a job like:

    #!/bin/bash
    #SBATCH --account=def-someuser
    #SBATCH --gres=gpu:1
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task=32
    #SBATCH --mem=0
    #SBATCH --time=5:00:00

    module load parabricks/2.5.0

    DATA_DIR=/path/to/data
    OUT_DIR=/path/to/output
    pbrun germline \
          --ref ${DATA_DIR}/Homo_sapiens_assembly38.fa \
          --in-fq ${DATA_DIR}/some_1.fastq ${DATA_DIR}/some_2.fastq \
          --knownSites ${DATA_DIR}/dbsnp_146.hg38.vcf.gz \
          --tmp-dir ${SLURM_TMPDIR}/ \
          --out-bam ${OUT_DIR}/output.bam \
          --out-variants ${OUT_DIR}/output.vcf \
          --out-recal-file ${OUT_DIR}/report.txt

## Common issues {#common_issues}

### Almost immediate failure {#almost_immediate_failure}

If your first test fails right away, there might be a missing module or some environmental variable clash. To solve this try:

### Later failure {#later_failure}

Often Parabricks may not give you a clear traceback of the failure. This usually means that that you did not request enough memory. If you are reserving a full node already through `--nodes=1`, we suggest you also use all the memory in the node with `--mem=0`. Otherwise, make sure that your pipeline has enough memory to process your data.

## Hybrid usage {#hybrid_usage}

Parabricks uses both CPU and GPUs. During our tests, Parabricks used at least 10 CPUs, so we recommend to ask for at least that amount through `--cpus-per-task=10`

# References

[Parabricks Home](http://www.nvidia.com/parabricks)
