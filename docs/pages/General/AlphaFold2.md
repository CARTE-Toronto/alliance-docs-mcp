---
title: "AlphaFold2/en"
url: "https://docs.alliancecan.ca/wiki/AlphaFold2/en"
category: "General"
last_modified: "2024-11-22T13:54:29Z"
page_id: 20387
display_title: "AlphaFold2"
---

`<languages />`{=html}

[AlphaFold](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology) is a machine learning model for the prediction of protein folding.

This page discusses how to use AlphaFold v2.0, the version that was entered in CASP14 and published in Nature.

Source code and documentation for AlphaFold can be found at their [GitHub page](https://github.com/deepmind/alphafold). Any publication that discloses findings arising from use of this source code or the model parameters should [cite](https://github.com/deepmind/alphafold#citing-this-work) the [AlphaFold paper](https://doi.org/10.1038/s41586-021-03819-2).

## Available versions {#available_versions}

AlphaFold is available on our clusters as prebuilt Python packages (wheels). You can list available versions with `avail_wheels`.

## Installing AlphaFold in a Python virtual environment {#installing_alphafold_in_a_python_virtual_environment}

1\. Load AlphaFold dependencies.

As of July 2022, only Python 3.7 and 3.8 are supported.

2\. Create and activate a Python virtual environment.

3\. Install a specific version of AlphaFold and its Python dependencies. X.Y.Z }} where `X.Y.Z` is the exact desired version, for instance `2.2.4`. You can omit to specify the version in order to install the latest one available from the wheelhouse.

4\. Validate it.

5\. Freeze the environment and requirements set.

## Databases

Note that AlphaFold requires a set of databases.

The databases are available in `/cvmfs/bio.data.computecanada.ca/content/databases/Core/alphafold2_dbs/`.

AlphaFold databases on CVMFS undergo yearly updates. In January 2024, the database was updated and is accessible in folder `2024_01`. /cvmfs/bio.data.computecanada.ca/content/databases/Core/alphafold2_dbs/2024_01/ }}

You can also choose to download the databases locally into your `$SCRATCH` directory.

`<b>`{=html}Important:`</b>`{=html} The databases must live in the `$SCRATCH` directory.

`<tabs>`{=html} `<tab name="General">`{=html} 1. From a DTN or login node, create the data folder. \$SCRATCH/alphafold/data \|mkdir -p \$DOWNLOAD_DIR }}

2\. With your modules loaded and virtual environment activated, you can download the data.

Note that this step `<b>`{=html}cannot`</b>`{=html} be done from a compute node. It should be done on a data transfer node (DTN) on clusters that have them (see [Transferring data](https://docs.alliancecan.ca/Transferring_data "Transferring data"){.wikilink}). On clusters that have no DTN, use a login node instead. Since the download can take up to a full day, we suggest using a [terminal multiplexer](https://docs.alliancecan.ca/Prolonging_terminal_sessions#Terminal_multiplexers "terminal multiplexer"){.wikilink}. You may encounter a `Client_loop: send disconnect: Broken pipe` error message. See [Troubleshooting](https://docs.alliancecan.ca/AlphaFold#Broken_pipe_error_message "Troubleshooting"){.wikilink} below.

`</tab>`{=html}

`<tab name="Graham only">`{=html} 1. Set `DOWNLOAD_DIR`. /datashare/alphafold }}

`</tab>`{=html} `</tabs>`{=html}

Afterwards, the structure of your data should be similar to `<tabs>`{=html} `<tab name=2.3>`{=html}

`</tab>`{=html}

`<tab name=2.2>`{=html}

`</tab>`{=html} `</tabs>`{=html}

## Running AlphaFold {#running_alphafold}

Edit one of following submission scripts according to your needs. `<tabs>`{=html} `<tab name="2.3 on CPU">`{=html} `{{File
|name=alphafold-2.3-cpu.sh
|lang="bash"
|contents=
#!/bin/bash

#SBATCH --job-name=alphafold_run
#SBATCH --account=def-someprof    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=08:00:00           # adjust this to match the walltime of your job
#SBATCH --cpus-per-task=8         # a MAXIMUM of 8 core, AlphaFold has no benefit to use more
#SBATCH --mem=20G                 # adjust this according to the memory you need

# Load modules dependencies.
module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 cuda/11.4 cudnn/8.2.0 kalign/2.03 hmmer/3.2.1 openmm-alphafold/7.5.1 hh-suite/3.3.0 python/3.8

DOWNLOAD_DIR=$SCRATCH/alphafold/data   # set the appropriate path to your downloaded data
INPUT_DIR=$SCRATCH/alphafold/input     # set the appropriate path to your input data
OUTPUT_DIR=${SCRATCH}/alphafold/output # set the appropriate path to your output data

# Generate your virtual environment in $SLURM_TMPDIR.
virtualenv --no-download ${SLURM_TMPDIR}/env
source ${SLURM_TMPDIR}/env/bin/activate

# Install AlphaFold and its dependencies.
pip install --no-index --upgrade pip
pip install --no-index --requirement ~/alphafold-requirements.txt

# Edit with the proper arguments and run your commands.
# run_alphafold.py --help
run_alphafold.py \
   --fasta_paths=${INPUT_DIR}/YourSequence.fasta,${INPUT_DIR}/AnotherSequence.fasta \
   --output_dir=${OUTPUT_DIR} \
   --data_dir=${DOWNLOAD_DIR} \
   --db_preset=full_dbs \
   --model_preset=multimer \
   --bfd_database_path=${DOWNLOAD_DIR}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
   --mgnify_database_path=${DOWNLOAD_DIR}/mgnify/mgy_clusters_2022_05.fa \
   --pdb70_database_path=${DOWNLOAD_DIR}/pdb70/pdb70 \
   --template_mmcif_dir=${DOWNLOAD_DIR}/pdb_mmcif/mmcif_files \
   --obsolete_pdbs_path=${DOWNLOAD_DIR}/pdb_mmcif/obsolete.dat \
   --pdb_seqres_database_path=${DOWNLOAD_DIR}/pdb_seqres/pdb_seqres.txt \
   --uniprot_database_path=${DOWNLOAD_DIR}/uniprot/uniprot.fasta \
   --uniref30_database_path=${DOWNLOAD_DIR}/uniref30/UniRef30_2021_03 \
   --uniref90_database_path=${DOWNLOAD_DIR}/uniref90/uniref90.fasta \
   --hhblits_binary_path=${EBROOTHHMINSUITE}/bin/hhblits \
   --hhsearch_binary_path=${EBROOTHHMINSUITE}/bin/hhsearch \
   --jackhmmer_binary_path=${EBROOTHMMER}/bin/jackhmmer \
   --kalign_binary_path=${EBROOTKALIGN}/bin/kalign \
   --max_template_date=2022-01-01 \
   --use_gpu_relax=False
}}`{=mediawiki} `</tab>`{=html}

`<tab name="2.3 on GPU">`{=html} `{{File
|name=alphafold-2.3-gpu.sh
|lang="bash"
|contents=
#!/bin/bash

#SBATCH --job-name=alphafold_run
#SBATCH --account=def-someprof    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=08:00:00           # adjust this to match the walltime of your job
#SBATCH --cpus-per-task=8         # a MAXIMUM of 8 core, AlphaFold has no benefit to use more
#SBATCH --gres=gpu:1              # a GPU helps to accelerate the inference part only
#SBATCH --mem=20G                 # adjust this according to the memory you need

# Load modules dependencies.
module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 cuda/11.4 cudnn/8.2.0 kalign/2.03 hmmer/3.2.1 openmm-alphafold/7.5.1 hh-suite/3.3.0 python/3.8

DOWNLOAD_DIR=$SCRATCH/alphafold/data   # set the appropriate path to your downloaded data
INPUT_DIR=$SCRATCH/alphafold/input     # set the appropriate path to your input data
OUTPUT_DIR=${SCRATCH}/alphafold/output # set the appropriate path to your output data

# Generate your virtual environment in $SLURM_TMPDIR.
virtualenv --no-download ${SLURM_TMPDIR}/env
source ${SLURM_TMPDIR}/env/bin/activate

# Install AlphaFold and its dependencies.
pip install --no-index --upgrade pip
pip install --no-index --requirement ~/alphafold-requirements.txt

# Edit with the proper arguments and run your commands.
# run_alphafold.py --help
run_alphafold.py \
   --fasta_paths=${INPUT_DIR}/YourSequence.fasta,${INPUT_DIR}/AnotherSequence.fasta \
   --output_dir=${OUTPUT_DIR} \
   --data_dir=${DOWNLOAD_DIR} \
   --db_preset=full_dbs \
   --model_preset=multimer \
   --bfd_database_path=${DOWNLOAD_DIR}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
   --mgnify_database_path=${DOWNLOAD_DIR}/mgnify/mgy_clusters_2022_05.fa \
   --pdb70_database_path=${DOWNLOAD_DIR}/pdb70/pdb70 \
   --template_mmcif_dir=${DOWNLOAD_DIR}/pdb_mmcif/mmcif_files \
   --obsolete_pdbs_path=${DOWNLOAD_DIR}/pdb_mmcif/obsolete.dat \
   --pdb_seqres_database_path=${DOWNLOAD_DIR}/pdb_seqres/pdb_seqres.txt \
   --uniprot_database_path=${DOWNLOAD_DIR}/uniprot/uniprot.fasta \
   --uniref30_database_path=${DOWNLOAD_DIR}/uniref30/UniRef30_2021_03 \
   --uniref90_database_path=${DOWNLOAD_DIR}/uniref90/uniref90.fasta \
   --hhblits_binary_path=${EBROOTHHMINSUITE}/bin/hhblits \
   --hhsearch_binary_path=${EBROOTHHMINSUITE}/bin/hhsearch \
   --jackhmmer_binary_path=${EBROOTHMMER}/bin/jackhmmer \
   --kalign_binary_path=${EBROOTKALIGN}/bin/kalign \
   --max_template_date=2022-01-01 \
   --use_gpu_relax=True
}}`{=mediawiki} `</tab>`{=html}

`<tab name="2.2 on CPU">`{=html} ``{{File
|name=alphafold-cpu.sh
|lang="bash"
|contents=
#!/bin/bash

#SBATCH --job-name=alphafold_run
#SBATCH --account=def-someprof    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=08:00:00           # adjust this to match the walltime of your job
#SBATCH --cpus-per-task=8         # a MAXIMUM of 8 core, AlphaFold has no benefit to use more
#SBATCH --mem=20G                 # adjust this according to the memory you need

# Load modules dependencies.
module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 cuda/11.4 cudnn/8.2.0 kalign/2.03 hmmer/3.2.1 openmm-alphafold/7.5.1 hh-suite/3.3.0 python/3.8

DOWNLOAD_DIR=$SCRATCH/alphafold/data   # set the appropriate path to your downloaded data
INPUT_DIR=$SCRATCH/alphafold/input     # set the appropriate path to your input data
OUTPUT_DIR=${SCRATCH}/alphafold/output # set the appropriate path to your output data

# Generate your virtual environment in $SLURM_TMPDIR.
virtualenv --no-download ${SLURM_TMPDIR}/env
source ${SLURM_TMPDIR}/env/bin/activate

# Install AlphaFold and its dependencies.
pip install --no-index --upgrade pip
pip install --no-index --requirement ~/alphafold-requirements.txt

# Edit with the proper arguments and run your commands.
# Note that the `--uniclust30_database_path` option below was renamed to
# `--uniref30_database_path` in 2.3.
# run_alphafold.py --help
run_alphafold.py \
   --fasta_paths=${INPUT_DIR}/YourSequence.fasta,${INPUT_DIR}/AnotherSequence.fasta \
   --output_dir=${OUTPUT_DIR} \
   --data_dir=${DOWNLOAD_DIR} \
   --model_preset=monomer_casp14 \
   --bfd_database_path=${DOWNLOAD_DIR}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
   --mgnify_database_path=${DOWNLOAD_DIR}/mgnify/mgy_clusters_2018_12.fa \
   --pdb70_database_path=${DOWNLOAD_DIR}/pdb70/pdb70 \
   --template_mmcif_dir=${DOWNLOAD_DIR}/pdb_mmcif/mmcif_files \
   --obsolete_pdbs_path=${DOWNLOAD_DIR}/pdb_mmcif/obsolete.dat \
   --uniclust30_database_path=${DOWNLOAD_DIR}/uniclust30/uniclust30_2018_08/uniclust30_2018_08  \
   --uniref90_database_path=${DOWNLOAD_DIR}/uniref90/uniref90.fasta  \
   --hhblits_binary_path=${EBROOTHHMINSUITE}/bin/hhblits \
   --hhsearch_binary_path=${EBROOTHHMINSUITE}/bin/hhsearch \
   --jackhmmer_binary_path=${EBROOTHMMER}/bin/jackhmmer \
   --kalign_binary_path=${EBROOTKALIGN}/bin/kalign \
   --max_template_date=2020-05-14 \
   --use_gpu_relax=False
}}``{=mediawiki} `</tab>`{=html}

`<tab name="2.2 on GPU">`{=html} ``{{File
|name=alphafold-gpu.sh
|lang="bash"
|contents=
#!/bin/bash

#SBATCH --job-name=alphafold_run
#SBATCH --account=def-someprof    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=08:00:00           # adjust this to match the walltime of your job
#SBATCH --gres=gpu:1              # a GPU helps to accelerate the inference part only
#SBATCH --cpus-per-task=8         # a MAXIMUM of 8 core, AlphaFold has no benefit to use more
#SBATCH --mem=20G                 # adjust this according to the memory you need

# Load modules dependencies.
module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 cuda/11.4 cudnn/8.2.0 kalign/2.03 hmmer/3.2.1 openmm-alphafold/7.5.1 hh-suite/3.3.0 python/3.8

DOWNLOAD_DIR=$SCRATCH/alphafold/data   # set the appropriate path to your downloaded data
INPUT_DIR=$SCRATCH/alphafold/input     # set the appropriate path to your input data
OUTPUT_DIR=${SCRATCH}/alphafold/output # set the appropriate path to your output data

# Generate your virtual environment in $SLURM_TMPDIR.
virtualenv --no-download ${SLURM_TMPDIR}/env
source ${SLURM_TMPDIR}/env/bin/activate

# Install AlphaFold  and its dependencies.
pip install --no-index --upgrade pip
pip install --no-index --requirement ~/alphafold-requirements.txt

# Edit with the proper arguments and run your commands.
# Note that the `--uniclust30_database_path` option below was renamed to
# `--uniref30_database_path` in 2.3.
# run_alphafold.py --help
run_alphafold.py \
   --fasta_paths=${INPUT_DIR}/YourSequence.fasta,${INPUT_DIR}/AnotherSequence.fasta \
   --output_dir=${OUTPUT_DIR} \
   --data_dir=${DOWNLOAD_DIR} \
   --model_preset=monomer_casp14 \
   --bfd_database_path=${DOWNLOAD_DIR}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
   --mgnify_database_path=${DOWNLOAD_DIR}/mgnify/mgy_clusters_2018_12.fa \
   --pdb70_database_path=${DOWNLOAD_DIR}/pdb70/pdb70 \
   --template_mmcif_dir=${DOWNLOAD_DIR}/pdb_mmcif/mmcif_files \
   --obsolete_pdbs_path=${DOWNLOAD_DIR}/pdb_mmcif/obsolete.dat \
   --uniclust30_database_path=${DOWNLOAD_DIR}/uniclust30/uniclust30_2018_08/uniclust30_2018_08  \
   --uniref90_database_path=${DOWNLOAD_DIR}/uniref90/uniref90.fasta  \
   --hhblits_binary_path=${EBROOTHHMINSUITE}/bin/hhblits \
   --hhsearch_binary_path=${EBROOTHHMINSUITE}/bin/hhsearch \
   --jackhmmer_binary_path=${EBROOTHMMER}/bin/jackhmmer \
   --kalign_binary_path=${EBROOTKALIGN}/bin/kalign \
   --max_template_date=2020-05-14 \
   --use_gpu_relax=True
}}``{=mediawiki} `</tab>`{=html} `</tabs>`{=html}

Then, submit the job to the scheduler.

## Troubleshooting

### Broken pipe error message {#broken_pipe_error_message}

When downloading the database, you may encounter a `Client_loop: send disconnect: Broken pipe` error message. It is hard to find the exact cause for this error message. It could be as simple as an unusually high number of users working on the login node, leaving less space for you to upload data.

- One solution is to use a [terminal multiplexer](https://docs.alliancecan.ca/Prolonging_terminal_sessions#Terminal_multiplexers "terminal multiplexer"){.wikilink}. Note that you could still encounter this error message but less are the chances.

<!-- -->

- A second solution is to use the database that is already present on the cluster. `/cvmfs/bio.data.computecanada.ca/content/databases/Core/alphafold2_dbs/2023_07/`.

<!-- -->

- Another option is to download the full database in sections. To have access to the different download scripts, after loading the module and activated your virtual environment, you simply enter `download_` in your terminal and tap twice on the `tab` keyboard key to visualize all the scripts that are available. You can manually download sections of the database by using the available script, as for instance `download_pdb.sh`.
