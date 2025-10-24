---
title: "CheckV/en"
url: "https://docs.alliancecan.ca/wiki/CheckV/en"
category: "General"
last_modified: "2023-11-27T20:11:58Z"
page_id: 24665
display_title: "CheckV/en"
---

`<languages />`{=html}

`<translate>`{=html}

CheckV is a fully automated command-line pipeline for assessing the quality of single-contig viral genomes, including identification of host contamination for integrated proviruses, estimating completeness for genome fragments, and identification of closed genomes. [CheckV](https://bitbucket.org/berkeleylab/checkv/src/master/) [PyPI-CheckV](https://pypi.org/project/checkv/)

Here is a demo explaining all the steps you should take when you want to use a software on the clusters. We will use CheckV as an example, but note that the point is for you to translate all these steps for the use of any other software.

## How to find if a software is available on the clusters? {#how_to_find_if_a_software_is_available_on_the_clusters}

### Modules

You will find all information about available software here [Available software](https://docs.alliancecan.ca/Available_software "Available software"){.wikilink}. In short, some of the softwares are available by loading the appropriate module.

To find the module you do:

`module spider nameOfYourSoftware`

You can also do the search by putting the name in between double quotes \"\" if you do not have the full name. Note that this search is not sensitive to the case so you should get the same output with uppercase and lowercase or a mix of them.

You also have the possibility to add the version number after the name to get more details about some modules you might need to load before and/or together with your software, we name them dependencies.

`module spider nameOfYourSoftware/10.2`

### Python packages {#python_packages}

In our example, we would not get any output for CheckV because it is a python wheel.

Python modules are provided as binary wheels [Available Python wheels](https://docs.alliancecan.ca/Available_Python_wheels "Available Python wheels"){.wikilink}.

You can find them by typing: `avail_wheels CheckV`.

You can apply the same search tricks as the module for double quotes and it is also insensitive to the case. You can add `--all-version` to list all the available versions.

## What do I need to do if the software I want to use is not available? {#what_do_i_need_to_do_if_the_software_i_want_to_use_is_not_available}

First steps would be to have a look at the documentation of the software. You can easily find the software development page that is often based on a github repository and follow the installation steps. Note that you **cannot** use Conda environment on the clusters [Anaconda](https://docs.alliancecan.ca/Anaconda "Anaconda"){.wikilink}. We have a wiki page that explains how you can install it locally in your account [Installing software in your home directory](https://docs.alliancecan.ca/Installing_software_in_your_home_directory "Installing software in your home directory"){.wikilink} or you can email the [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink} to get some help to either install it in your account or in the clusters.

For python wheels, you can search them on [PyPI](https://pypi.org/) website which is a collection of wheels made available for everyone. We will get in more details in the following section but you can install them in your virtual environment with this command: `pip install nameOfTheWheel`.

You can also contact us to add your preferred wheel on the wheelhouse as this command is not installing the wheel from our wheelhouse but from the web. To install it from our wheelhouse you need to add `--no-index` parameter. `pip install nameOfTheWheel --no-index`

## Installation

**1. Load the necessary modules.**

As mentioned in section [CheckV#How to find if a software is available on the clusters?](https://docs.alliancecan.ca/CheckV#How_to_find_if_a_software_is_available_on_the_clusters? "CheckV#How to find if a software is available on the clusters?"){.wikilink}, you can find the dependencies that are necessary to load before you load your software by looking at a specific version with `module spider nameOfYourSoftware/10.2`

There could also be other dependencies, you usually find them on the software development page. Note that you would need to go though section [CheckV#How to find if a software is available on the clusters?](https://docs.alliancecan.ca/CheckV#How_to_find_if_a_software_is_available_on_the_clusters? "CheckV#How to find if a software is available on the clusters?"){.wikilink} for all dependencies to find if they are present on the clusters.

**2. Create and activate the virtual environment.**

**3. You should also upgrade `pip` in the environment.**

This step is important if you are using python version \< then 3.10.2.

**4. Install the wheel and its dependencies (if you have any).**

`   4.1 A wheel from the wheelhouse (prefered choice):`

`   4.2 A wheel from the web. Note that if you install a wheel from the web inside your virtual environment you will not be able to use a requirement file. You would need to do option 4.3 as an alternative. `

`   4.3 If you want to use a wheel from the web and also use a requirement file, you would need to do the following command outside the virtual environment.`

**5. Validate it.**

Freeze the environment and requirements set. For requirements text file usage, have a look at the bash submission script described in point number {}. Remember that you can use a requirement file only with installation option 4.1 and 4.3.

## Datasets

**1. Download the database**

You must pre-download the database before submitting your job. For intensive read/write operations on large files, scratch storage space is the best choice. This is why we usually recommend downloading databases in your scratch.

Some users may wish to update the database using their own complete genomes:

Some users may wish to download a specific database version. See [https://portal.nersc.gov/CheckV/](https://portal.nersc.gov/CheckV/ "https://portal.nersc.gov/CheckV/"){.wikilink} for an archive of all previous database versions. If you go this route then you\'ll need to build the DIAMOND database manually:

**2. Download a sequence test** Some software will give access to a data set for you to test the software. You can look if anything is available on the web or the github repository. For CheckV, the data set is available here [1](https://bitbucket.org/berkeleylab/checkv/src/master/test/). You can download it with this command:

## Usage

### Job submission {#job_submission}

#### Interactive session {#interactive_session}

First step for running your job : use an interactive session.

Still need to demystify #SBATCH parameter? If you need to refresh your SBATCH parameter knowledge we recommend having a look at the [Slurm SBATCH command page](https://slurm.schedmd.com/sbatch.html) and the [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink} wiki page.

To learn more about interactive job you can have a look at the wiki page [Running jobs#Interactive job](https://docs.alliancecan.ca/Running_jobs#Interactive_job "Running jobs#Interactive job"){.wikilink}.

**1. Gather information on the command line and the software.**

The first thing you need to do is to analyze the proposed command line and look in the help menu if there is any information about threading or parameters to help you set up an HPC (High performance computing) usage.

In our case, here is the command line proposed for a full pipeline analysis:

In this case, you should be intrigued in finding what the `-t` parameter is doing. To have access to the help menu for end_to_end program:

First thing to look at is the usage. This tells us in which order we need to provide the arguments. Also, we have the details about parameter `-t` that tells us that CheckV is using Prodigal and DIAMOND that are programs written with multithreaded code so we can modify this parameter to adjust the number of threads.

**2. Launch the salloc command.**

For CheckV, here is the command line you should ask as a first trial:

1:0:0 \--mem2G \--cpus-per-task16 \--accountdef-someuser}}

2.1 Time First of all, the use of an interactive job is not to run a script for 24h. It is to explore a command line or an environment tool for software development in terms of debugging or compiling. So if you wish to work for 1h or 2h this would be your number.

2.2 Memory The objective is to start low and increase if needed. Here you can start with 2G.

2.3 CPU Same as the memory, here is to go from lower to higher. As suggested in the proposed command line in the CheckV software, we can ask for 16 tasks. Load the appropriate modules Note that it is important to do this step before activating your virtual environment.

**3. Load the necessary modules.**

**4. Activate the virtual environment**

**5. Execute your command**

Here we are using the data set downloaded in section \[CheckV#Download a sequence test\]. As mentioned before, it is a good habit to look into the git repository if any data tests are available and use it in your first attempts to run your software. `{{Commands
|checkv end_to_end test_sequences.fna $SCRATCH/checkv_output -t ${SLURM_CPUS_PER_TASK:-1} -d $SCRATCH/checkv-db-v1.5    
}}`{=mediawiki} **6. Gather information about the run.** The `checkv end_to_end` command line had 4 different analysis outputs. The execution time was less than a minute for all sections together with a peak of memory just under 2GB. This is important information because we are going to use those time and memory parameters in the bash submission script to launch the command line on the compute nodes.

#### Compute node job submission {#compute_node_job_submission}

How to set up a bash submission script and submit it on the compute node?

**1. Compute node submission** Here you have an example of a submission script. You can submit as so: `sbatch run_CheckV.sh`.

```{=mediawiki}
{{File
  |name=run_CheckV.sh
  |lang="bash"
  |contents=

#!/bin/bash

#SBATCH --job-name=CheckV_run
#SBATCH --account=def-someprof    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=00:20:00           # adjust this to match the walltime of your job
#SBATCH --cpus-per-task=16        # adjust depending on Prodigal and DIAMOND usage
#SBATCH --mem=2G                  # adjust this according to the memory you need

# Load modules dependencies.
module load gcc hmmer/3.3.2 prodigal-gv/2.6.3 diamond/2.0.4 python/3.10

# Generate your virtual environment in $SLURM_TMPDIR.
virtualenv --no-download ${SLURM_TMPDIR}/env
source ${SLURM_TMPDIR}/env/bin/activate

# Install CheckV and its dependencies.
pip install --no-index --upgrade pip
pip install --no-index --requirement ~/checkv-1.0.1-requirements.txt

# Edit with the proper arguments, run your commands.
checkv end_to_end test_sequences.fna $SCRATCH/checkv_output -t ${SLURM_CPUS_PER_TASK:-1} -d $SCRATCH/checkv-db-v1.5

}}
```
### Useful commands to adjust resources. {#useful_commands_to_adjust_resources.}

To have the whole picture about commands involved in job monitoring please have a look at this wiki page: \[Running_jobs#Monitoring_jobs Monitoring jobs\].

**1. Seff command.**

The seff command is a post run command. It uses the slurm job number as parameter, `seff 42760046`. You will be able to see the time, the CPU efficiency and the memory efficiency. Depending on the percentage noted in the output you will be able to adjust future runs so you are not going to waste any ressources.

**2. Slurm output file.** Have a look at the slurm output file `slurm-[job_number].out`.

The slurm output file can give you useful information about the run.

`less slurm-[job_number].out`

`</translate>`{=html}
