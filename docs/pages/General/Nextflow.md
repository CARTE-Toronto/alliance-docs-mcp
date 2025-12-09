---
title: "Nextflow/en"
url: "https://docs.alliancecan.ca/wiki/Nextflow/en"
category: "General"
last_modified: "2025-11-24T22:04:01Z"
page_id: 22694
display_title: "Nextflow"
---

Nextflow is software for running reproducible scientific workflows. The term Nextflow is used to describe both the domain-specific-language (DSL) the pipelines are written in, and the software used to interpret those workflows.

== Usage ==
On our systems, Nextflow is provided as a module you can load with module load nextflow.

While you can build your own workflow, you can also rely on the published nf-core pipelines. We will describe here a simple configuration that will let you run nf-core pipelines on our systems and help you to configure Nextflow properly for your own pipelines.

Our example uses the nf-core/rnaseq pipeline and has five steps:

* Step 1: Set up the configuration file.
* Step 2: Install nf-core.
* Step 3: Download the container image and pipeline.
* Step 4: Prepare the inputs.
* Step 5: Create a job script.

==== Step 1: Set up the configuration file ====

Obtain a configuration file for Alliance clusters from nf-core
and place it in ~/.nextflow/config like so:

curl -o ~/.nextflow/config https://raw.githubusercontent.com/nf-core/configs/refs/heads/master/conf/alliance_canada.config

Set the $SLURM_ACCOUNT environment variable to an account name you can use.  It should look like "def-pname". You may choose to set this in your ~/.bashrc file, for your convenience.

export SLURM_ACCOUNT=def-pname

This configuration contains profiles for Fir, Narval, Nibi, Rorqual, and Trillium. If you use this configuration file at Fir you must load the profile using the -profile fir flag to the nextflow command.
At other sites the appropriate profile is automatically selected based on the host name. It ensures that there are no more than 100 jobs in the Slurm queue and that no more than 60 jobs are submitted per minute. It contains cluster-specific information that Nextflow needs, for example that Rorqual machines have 192 cores and 750 GB of RAM with a maximum time limit of one week (168 hours).

We discourage you from running nf-core pipelines or any other generic Nextflow pipeline on Trillium. We recommend running a pipeline on Trillium only if it was designed specifically for Trillium.

The configuration is linked to the system you are running on, but it is also related to the pipeline itself. In this rnaseq example cpu = 1 is the default value, but steps in the pipeline can have more than that. This can get quite complicated and labels in the nf-core-rnaseq_3.21.0/3_21_0/conf/base.config file are used by the pipeline to identify steps with non-default configurations. We do not cover this more advanced topic here, but note that tweaking these labels could make a big difference in the queuing and execution time of your pipeline.

==== Step 2. Install nf-core ====

To use nf-core pipelines on an Alliance cluster, the pipelines must be downloaded on a login node because some clusters do not allow internet access from the compute nodes.
Run the following on a login node to install nf-core.

module purge
module load python/3.11
module load rust         # New nf-core installations will error out if rust hasn't been loaded
module load postgresql   # Python modules which list psycopg2 as a dependency may crash without postgresql here.
python -m venv nf-core-env
source nf-core-env/bin/activate
python -m pip install nf_core==2.13

We use pip to install a Python package to help with the setup. The nf-core tools can be slow to install; this step may take several minutes.

==== Step 3. Download the container images and the pipeline ====
Set the name of the pipeline to be tested, and load Nextflow and the container utility Apptainer.
Nextflow integrates well with Apptainer.
As noted above, we are using hte rna-seq pipeline as an example.

export NFCORE_PL=rnaseq
export PL_VERSION=3.21.0
module load nextflow
module load apptainer

Create a directory to use as a cache:

mkdir /project//NXF_SINGULARITY_CACHEDIR
export NXF_SINGULARITY_CACHEDIR=/project//NXF_SINGULARITY_CACHEDIR

Nextflow will store container images in the directory pointed to by $NXF_SINGULARITY_CACHEDIR.
"Singularity" was a predecessor to "Apptainer" so the name of the variable still reflects that.
Workflow images tend to be big, so do not store them in your $HOME space because it has a small quota.
Instead, store them in /project space.

You should share this folder with other members of your group who are planning to use Nextflow with Apptainer, in order to reduce duplication and save space.
Also, you may add the export command to your ~/.bashrc as a convenience.

Run the following command to download the rnaseq pipeline and container images.

cd ~/scratch
mkdir -p nf-test && cd nf-test
nf-core download --container-cache-utilisation amend --container-system singularity --compress none -r ${PL_VERSION}  -p 6  ${NFCORE_PL}

Type "Y" when you see Include the nf-core's default institutional configuration files into the download? (Y/n)

IMPORTANT!: This workflow downloads two components of  rnaseq:

# Container image files go into $NXF_SINGULARITY_CACHEDIR
# Pipeline files go into ~/scratch/nf-test/nf-core-${NFCORE_PL}_${PL_VERSION} folder with the version number X_X_X. In this example the pipeline is stored at ~/scratch/nf-test/nf-core-rnaseq_3.21.0/3_21_0. Please note that you have to include this nf-core-rnaseq_3.21.0/3_21_0 folder name when calling nextflow run in your job script (see Step 5 below).

When the pipeline is launched, Nextflow will look at the nextflow.config file in the working directory and also at ~/.nextflow/config (if it exists) in your home to control how to run the workflow.
The nf-core pipelines all have a default configuration, a test configuration, and container configurations (singularity, podman, etc).

==== Step 4. Prepare the input files ====

Nextflow uses sequence files and a sample sheet as inputs.
To download the sequence files needed for our 'rnaseq' example, run the following:

cd ~/scratch/nf-test
mkdir -p input && cd input
wget https://raw.githubusercontent.com/nf-core/test-datasets/rnaseq/testdata/GSE110004/SRR6357070_1.fastq.gz
wget https://raw.githubusercontent.com/nf-core/test-datasets/rnaseq/testdata/GSE110004/SRR6357070_2.fastq.gz
wget https://raw.githubusercontent.com/nf-core/test-datasets/rnaseq/testdata/GSE110004/SRR6357071_1.fastq.gz
wget https://raw.githubusercontent.com/nf-core/test-datasets/rnaseq/testdata/GSE110004/SRR6357071_2.fastq.gz
wget https://raw.githubusercontent.com/nf-core/test-datasets/rnaseq/testdata/GSE110004/SRR6357072_1.fastq.gz
wget https://raw.githubusercontent.com/nf-core/test-datasets/rnaseq/testdata/GSE110004/SRR6357072_2.fastq.gz
wget https://raw.githubusercontent.com/nf-core/test-datasets/rnaseq/testdata/GSE110004/SRR6357073_1.fastq.gz
wget https://raw.githubusercontent.com/nf-core/test-datasets/rnaseq/testdata/GSE110004/SRR6357074_1.fastq.gz
wget https://raw.githubusercontent.com/nf-core/test-datasets/rnaseq/testdata/GSE110004/SRR6357075_1.fastq.gz
wget https://raw.githubusercontent.com/nf-core/test-datasets/rnaseq/testdata/GSE110004/SRR6357076_1.fastq.gz
wget https://raw.githubusercontent.com/nf-core/test-datasets/rnaseq/testdata/GSE110004/SRR6357076_2.fastq.gz

To prepare a sample sheet, copy and paste the following into ~/scratch/nf-test/samplesheet.csv
and then change the username to your own::

sample,fastq_1,fastq_2,strandedness
WT_REP1,/home//scratch/nf-test/input/SRR6357070_1.fastq.gz,/home//scratch/nf-test/input/SRR6357070_2.fastq.gz,reverse
WT_REP1,/home//scratch/nf-test/input/SRR6357071_1.fastq.gz,/home//scratch/nf-test/input/SRR6357071_2.fastq.gz,reverse
WT_REP2,/home//scratch/nf-test/input/SRR6357072_1.fastq.gz,/home//scratch/nf-test/input/SRR6357072_2.fastq.gz,reverse
RAP1_UNINDUCED_REP1,/home//scratch/nf-test/input/SRR6357073_1.fastq.gz,,reverse
RAP1_UNINDUCED_REP2,/home//scratch/nf-test/input/SRR6357074_1.fastq.gz,,reverse
RAP1_UNINDUCED_REP2,/home//scratch/nf-test/input/SRR6357075_1.fastq.gz,,reverse
RAP1_IAA_30M_REP1,/home//scratch/nf-test/input/SRR6357076_1.fastq.gz,/home//scratch/nf-test/input/SRR6357076_2.fastq.gz,reverse

You can of course use your own data if you prefer.
Read more here about the rnaseq example and sample sheets.

====Step 5. Create a job script====

Here is the example job script for use on Fir.
Adapt the script to use the correct:
* pipeline (NFCORE_PL) and version (PL_VERSION, FD_VERSION)
* Apptainer cache path (NXF_SINGULARITY_CACHEDIR)
* Slurm account (SLURM_ACCOUNT)
* cluster (-profile ...,fir)
* paths for --input and --output

#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load python/3.11
source nf-core-env/bin/activate
module load apptainer
module load nextflow
export NFCORE_PL=rnaseq
export PL_VERSION=3.21.0
export FD_VERSION=3_21_0
export NXF_SINGULARITY_CACHEDIR=/project//NXF_SINGULARITY_CACHEDIR
export SLURM_ACCOUNT=def-pname

nextflow run nf-core-${NFCORE_PL}_${PL_VERSION}/${FD_VERSION}/ \
 -profile test,singularity,fir \
 --input ~/scratch/nf-test/input/samplesheet.csv --outdir ~/scratch/nf-test/output

Save the job script in ~/scratch/nf-test/nextflow_test.sh,
then submit it with sbatch nextflow_test.sh to launch the test run.

So now you have started Nextflow on the compute node. This process sends jobs to Slurm when they are ready to be processed.

You can see the progression of the pipeline from the log file nextflow_test..out . You can also have a look at the jobs in the Slurm queue with sq or squeue -u $USER.

To learn more about configurations and profiles in Nextflow, see:
* "Configuration"
* "Pipeline configuration"

== Known issues ==

Note that Nextflow is mainly written in Java which tends to use a lot of virtual memory. On some clusters, this may be a problem when running from a login node.

Be careful if you have an AWS configuration in your ~/.aws directory, as Nextflow might complain that it can't download the pipeline test dataset with your default id.

==== "unable to create native thread" ====

The following error has been observed:

java.lang.OutOfMemoryError: unable to create native thread: possibly out of memory or process/resource limits reached
[error][gc,task] GC Failed to create worker thread

We believe this is due to Java trying to create threads to match the number of physical cores on a machine.
Setting export NXF_OPTS='-XX:ActiveProcessorCount=1' when executing nextflow is reported to solve the problem.

==== SIGBUS ====
Some users have reported getting a SIGBUS error from the Nextflow main process.
We suspect this is connected with these Nextflow issues:
 * https://github.com/nextflow-io/nextflow/issues/842
 * https://github.com/nextflow-io/nextflow/issues/2774
Setting the environment variable NXF_OPTS="-Dleveldb.mmap=false" when executing nextflow is reported to solve the problem.