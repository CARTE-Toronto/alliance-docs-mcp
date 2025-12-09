---
title: "BUSCO/en"
url: "https://docs.alliancecan.ca/wiki/BUSCO/en"
category: "General"
last_modified: "2024-05-01T23:07:01Z"
page_id: 9850
display_title: "BUSCO"
---

`<languages />`{=html}

BUSCO (`<i>`{=html}Benchmarking sets of Universal Single-Copy Orthologs`</i>`{=html}) is an application for assessing genome assembly and annotation completeness.

For more information, see the [user manual](https://busco.ezlab.org/busco_userguide.html).

## Available versions {#available_versions}

Recent versions are available as wheels. Older versions are available as a module; please see the [Modules](https://docs.alliancecan.ca/#Modules "Modules"){.wikilink} section below.

To see the latest available version, run

## Python wheel {#python_wheel}

### Installation

`<b>`{=html}1.`</b>`{=html} Load the necessary modules.

`<b>`{=html}2.`</b>`{=html} Create the virtual environment.

`<b>`{=html}3.`</b>`{=html} Install the wheel and its dependencies. 1.81 pandas2.1.0 busco5.5.0 }}

`<b>`{=html}4.`</b>`{=html} Validate the installation.

`<b>`{=html}5.`</b>`{=html} Freeze the environment and requirements set. To use the requirements text file, see the `<i>`{=html}bash`</i>`{=html} submission script shown at point 8.

### Usage

#### Datasets

`<b>`{=html}6.`</b>`{=html} You must pre-download any datasets from [BUSCO data](https://busco-data.ezlab.org/v5/data/) before submitting your job.

You can access the available datasets in your terminal by typing `busco --list-datasets`.

You have `<b>`{=html}two`</b>`{=html} options to download datasets:\
\*use the `busco` command,

- use the `wget` command.

##### `<b>`{=html}6.1`</b>`{=html} Using the `busco` command {#using_the_busco_command}

This is the preferred option. Type this command in your working directory to download a particular dataset, for example

It is also possible to do a bulk download by replacing the dataset name by the following arguments: `all`, `prokaryota`, `eukaryota`, or `virus`, for example

This will

:   

    :   1\. create a BUSCO directory hierarchy for the datasets,
    :   2\. download the appropriate datasets,
    :   3\. decompress the file(s),
    :   4\. if you download multiple files, they will all be automatically added to the lineages directory.

The hierarchy will look like this:

> - busco_downloads/
>
> ::\* information/
>
> :   
>
>     :   lineages_list.2021-12-14.txt
>
> ::\* lineages/
>
> :   
>
>     :   bacteria_odb10
>
> <!-- -->
>
> :   
>
>     :   actinobacteria_class_odb10
>
> <!-- -->
>
> :   
>
>     :   actinobacteria_phylum_odb10
>
> ::\* placement_files/
>
> :   
>
>     :   list_of_reference_markers.archaea_odb10.2019-12-16.txt

Doing so, all your lineage files should be in `<b>`{=html}busco_downloads/lineages/`</b>`{=html}. When referring to `--download_path busco_downloads/` in the BUSCO command line, it will know where to find the lineage dataset argument `--lineage_dataset bacteria_odb10`. If the `<i>`{=html}busco_download `</i>`{=html} directory is not in your working directory, you will need to provide the full path.

##### `<b>`{=html}6.2`</b>`{=html} Using the `wget` command {#using_the_wget_command}

All files must be decompressed with `tar -xvf file.tar.gz`.

#### Test

`<b>`{=html}7.`</b>`{=html} Download a genome file.

`<b>`{=html}8.`</b>`{=html} Run.

Command to run a single genome:

```{=mediawiki}
{{Command|busco --offline --in genome.fna --out TEST --lineage_dataset bacteria_odb10 --mode genome --cpu ${SLURM_CPUS_PER_TASK:-1} --download_path busco_download/}}
```
Command to run multiple genomes that would be saved in the genome directory (in this example, the `<i>`{=html}genome/`</i>`{=html} folder would need to be in the current directory; otherwise, you need to provide the full path):

```{=mediawiki}
{{Command|busco --offline --in genome/ --out TEST --lineage_dataset bacteria_odb10 --mode genome --cpu ${SLURM_CPUS_PER_TASK:-1} --download_path busco_download/}}
```
The single genome command should take less than 60 seconds to complete. Production runs which take longer must be submitted to the [scheduler](https://docs.alliancecan.ca/Running_jobs "scheduler"){.wikilink}.

##### BUSCO tips {#busco_tips}

Specify `--in genome.fna` for single file analysis.

Specify `--in genome/` for multiple files analysis.

##### Slurm tips {#slurm_tips}

Specify `--offline` to avoid using the internet.

Specify `--cpu` to `$SLURM_CPUS_PER_TASK` in your job submission script to use the number of CPUs allocated.

Specify `--restart` to restart from a partial run.

#### Job submission {#job_submission}

Here you have an example of a submission script. You can submit as so: `sbatch run_busco.sh`.

```{=mediawiki}
{{File
  |name=run_busco.sh
  |lang="bash"
  |contents=

#!/bin/bash

#SBATCH --job-name=busco9_run
#SBATCH --account=def-someprof    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=01:00:00           # adjust this to match the walltime of your job
#SBATCH --cpus-per-task=8         # adjust depending on the size of the genome(s)/protein(s)/transcriptome(s)
#SBATCH --mem=20G                 # adjust this according to the memory you need

# Load modules dependencies.
module load StdEnv/2020 gcc/9.3.0 python/3.10 augustus/3.5.0 hmmer/3.3.2 blast+/2.13.0 metaeuk/6 prodigal/2.6.3 r/4.3.1 bbmap/38.86

# Generate your virtual environment in $SLURM_TMPDIR.
virtualenv --no-download ${SLURM_TMPDIR}/env
source ${SLURM_TMPDIR}/env/bin/activate

# Install busco and its dependencies.
pip install --no-index --upgrade pip
pip install --no-index --requirement ~/busco-requirements.txt

# Edit with the proper arguments, run your commands.
busco --offline --in genome.fna --out TEST --lineage_dataset bacteria_odb10 --mode genome --cpu ${SLURM_CPUS_PER_TASK:-1} --download_path busco_download/

}}
```
#### Augustus parameters {#augustus_parameters}

`<b>`{=html}9.`</b>`{=html} Advanced users may want to use Augustus parameters: `--augustus_parameters="--yourAugustusParameter"`.

- Copy the Augustus `<i>`{=html}config`</i>`{=html} directory to a writable location.

<!-- -->

- Make sure to define the `AUGUSTUS_CONFIG_PATH` environment variable.

\$HOME/augustus_config}}

#### SEPP parameters {#sepp_parameters}

`<b>`{=html}10.`</b>`{=html} To use SEPP parameters, you need to install SEPP locally in your virtual environment. This should be done from the login node.

`<b>`{=html}10.1.`</b>`{=html} Activate your BUSCO virtual environment.

`<b>`{=html}10.2.`</b>`{=html} Install DendroPy.

`<b>`{=html}10.3.`</b>`{=html} Install SEPP.

`<b>`{=html}10.4.`</b>`{=html} Validate the installation.

`<b>`{=html}10.5.`</b>`{=html} Because SEPP is installed locally, you cannot create the virtual environment as described in the previous submission script. To activate your local virtual environment, simply add the following command immediately under the line to load the module:

## Modules

`<b>`{=html}1.`</b>`{=html} Load the necessary modules.

This will also load modules for Augustus, BLAST+, HMMER and some other software packages that BUSCO relies upon.

`<b>`{=html}2.`</b>`{=html} Copy the configuration file.

or

`<b>`{=html}3.`</b>`{=html} Edit the configuration file. The locations of external tools are all specified in the last section, which is shown below:

`<b>`{=html}4.`</b>`{=html} Copy the Augustus `config` directory to a writable location.

`<b>`{=html}5.`</b>`{=html} Check that it runs.

\$HOME/busco_config.ini \|export AUGUSTUS_CONFIG_PATH\$HOME/augustus_config \|run_BUSCO.py \--in \$EBROOTBUSCO/sample_data/target.fa \--out TEST \--lineage_path \$EBROOTBUSCO/sample_data/example \--mode genome }}

The `run_BUSCO.py` command should take less than 60 seconds to complete. Production runs which take longer should be submitted to the [scheduler](https://docs.alliancecan.ca/Running_jobs "scheduler"){.wikilink}.

# Troubleshooting

## Cannot write to Augustus config path {#cannot_write_to_augustus_config_path}

Make sure you have copied the `<i>`{=html}config`</i>`{=html} directory to a writable location and exported the `AUGUSTUS_CONFIG_PATH` variable.
