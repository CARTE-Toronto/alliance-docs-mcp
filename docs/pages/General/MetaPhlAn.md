---
title: "MetaPhlAn/en"
url: "https://docs.alliancecan.ca/wiki/MetaPhlAn/en"
category: "General"
last_modified: "2022-11-11T17:17:59Z"
page_id: 21607
display_title: "MetaPhlAn"
---

MetaPhlAn is a "computational tool for profiling the composition of microbial communities (Bacteria, Archaea and Eukaryotes) from metagenomic shotgun sequencing data (i.e. not 16S) with species-level. With StrainPhlAn, it is possible to perform accurate strain-level microbial profiling", according to its GitHub repository. While the software stack on our clusters does contain modules for a couple of older versions (2.2.0 and 2.8) of this software, we now expect users to install recent versions using a  Python virtual environment.

For more information on how to use MetaPhlan, see their wiki

= Available wheels =
You can list available wheels using the avail_wheels command:

= Downloading databases =
Note that MetaPhlAn requires a set of databases to be downloaded into the $SCRATCH.

Important: The database must live in the $SCRATCH

Databases can be downloaded from Segatalab FTP .

1. From a login node, create the data folder:
$SCRATCH/metaphlan_databases
|mkdir -p $DB_DIR
|cd $DB_DIR
}}

2. Download the data:

Note that this step cannot be done from a compute node but must be done from a login node.

3. Extract the downloaded data, for example using an interactive job:
 --cpus-per-task2 --mem10G
}}
Untar and unzip the databases:

= Running MetaPhlAn =
Once the database files have been downloaded and extracted, you can submit a job. You may edit the following job submission script
according to your needs:
$SCRATCH/metaphlan_databases

# Generate your virtual environment in $SLURM_TMPDIR
virtualenv --no-download ${SLURM_TMPDIR}/env
source ${SLURM_TMPDIR}/env/bin/activate

# Install metaphlan and its dependencies
pip install --no-index --upgrade pip
pip install --no-index metaphlan==X.Y.Z  # EDIT: the required version here, e.g. 4.0.3

# Reuse the number of core allocated to our job from `--cpus-per-task=4`
# It is important to use --index and --bowtie2db so that MetaPhlAn can run inside the job
metaphlan metagenome.fastq --input_type fastq -o profiled_metagenome.txt --nproc $SLURM_CPUS_PER_TASK --index mpa_vJan21_CHOCOPhlAnSGB_202103 --bowtie2db $DB_DIR --bowtie2out metagenome.bowtie2.bz2
}}

Then submit the job to the scheduler: