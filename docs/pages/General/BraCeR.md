---
title: "BraCeR"
url: "https://docs.alliancecan.ca/wiki/BraCeR"
category: "General"
last_modified: "2024-07-15T19:11:42Z"
page_id: 20616
display_title: "BraCeR"
---

[BraCeR](https://github.com/Teichlab/bracer) - reconstruction of B cell receptor sequences from single-cell RNA-seq data.

## Installation

The follwing will install parts of BraCeR in the following locations:

- Content of the BraCeR Git-reposistory: `$HOME/bracer`
- Virtual Python environment: `$HOME/venv_bracer`
- Transcriptome databeses for Kallisto: `/scratch/$USER/GRCh38` and `/scratch/$USER/GRCm38`

However it could also be structured differently. Please refer to our [Storage and file management](https://docs.alliancecan.ca/Storage_and_file_management#Storage_types "Storage and file management"){.wikilink} page for a comparism of the different storeage types available.

`# BraCeR installation instructions as of August 2022.`\
`# Get bracer`\
`cd ~/`\
`git clone `[`https://github.com/Teichlab/bracer.git`](https://github.com/Teichlab/bracer.git)\
\
`# load modules`\
`module load StdEnv/2020`\
`module load python/3.8`\
`module load gcc/9.3.0 bowtie2/2.4.1 bowtie/1.3.0 trinity/2.14.0 samtools/1.15.1`\
`module load igblast/1.17.0   blast+/2.12.0    kallisto/0.46.1`\
`module load samtools/1.15.1  jellyfish/2.3.0`\
`module load salmon/1.7.0     fastqc/0.11.9`\
\
`# create a virtualenv for bracer and install dependencies`\
`virtualenv --no-download  ~/venv_bracer`\
`source ~/venv_bracer/bin/activate`\
`pip install --no-index biopython==1.77`\
`pip install --no-index -r ~/bracer/requirements.txt`\
`pip install --no-index graphviz`\
\
`# install bracer into venv`\
`cd ~/bracer/`\
`python setup.py install`\
\
`# install Trim Galore`\
`pusd $VIRTUAL_ENV/`\
`## download and extract Trim Galore`\
`wget `[`https://github.com/FelixKrueger/TrimGalore/archive/refs/tags/0.6.7.tar.gz`](https://github.com/FelixKrueger/TrimGalore/archive/refs/tags/0.6.7.tar.gz)` \`\
`    -O TrimGalore-0.6.7.tar.gz`\
`tar xzf TrimGalore-0.6.7.tar.gz`\
`# tweak the Perl "hashbang":`\
`sed -i 's&#!/usr/bin/perl&#!/usr/bin/env perl&'  TrimGalore-0.6.7/trim_galore`\
`# copy  the trim_galore Perl-script into the virtualenv's "bin" directory:`\
`cp  TrimGalore-0.6.7/trim_galore  $VIRTUAL_ENV/bin`\
`popd`\
\
`# Base transcriptomes for Kallisto`\
`# adapted `[`https://github.com/Teichlab/bracer#base-transcriptomes-for-kallisto`](https://github.com/Teichlab/bracer#base-transcriptomes-for-kallisto)\
`# to download from `[`https://www.gencodegenes.org/`](https://www.gencodegenes.org/)\
`# This downloads the latest releases 41 (Human) and M30 (mouse) respectively`\
`cd ~/scratch`\
`mkdir GRCh38`\
`cd GRCh38`\
`wget `[`https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.transcripts.fa.gz`](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.transcripts.fa.gz)\
`gunzip gencode.v41.transcripts.fa.gz`\
`python3 ~/bracer/docker_helper_files/gencode_parse.py gencode.v27.transcripts.fa`\
`cd ..`\
\
`mkdir GRCm38`\
`cd GRCm38`\
`wget `[`https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M30/gencode.vM30.transcripts.fa.gz`](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M30/gencode.vM30.transcripts.fa.gz)\
`gunzip gencode.vM30.transcripts.fa.gz`\
`python3 ~/bracer/docker_helper_files/gencode_parse.py gencode.vM30.transcripts.fa`\
`cd ~/`

## Running Bracer {#running_bracer}

Note: Make sure to replace USERNAME with your username in the `bracer.conf`:

### Bracer.conf

### Jobscript

```{=mediawiki}
{{File
  |name=bracer_job.sh
  |lang="bash"
  |contents=
#!/bin/bash
#SBATCH --time=0-00:30  # 30 minutes  (D-hh:mm)
#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=4000M

module load StdEnv/2020
module load gcc/9.3.0 bowtie2/2.4.1 bowtie/1.3.0 trinity/2.14.0 samtools/1.15.1
module load igblast/1.17.0   blast+/2.12.0    kallisto/0.46.1
module load samtools/1.15.1  jellyfish/2.3.0
module load salmon/1.7.0     fastqc/0.11.9

echo "---------------------------------------------------------------------"
module -w 80 list
echo "---------------------------------------------------------------------"

source ~/venv_bracer/bin/activate

bracer assemble  --ncores ${SLURM_CPUS_PER_TASK:-1} --config_file bracer.conf  \
       my_cell_name   ./my_output_directory/   file_001.fastq   file_002.fastq \
}}
```
