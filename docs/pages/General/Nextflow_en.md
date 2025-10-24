---
title: "Nextflow/en"
url: "https://docs.alliancecan.ca/wiki/Nextflow/en"
category: "General"
last_modified: "2025-10-17T16:43:51Z"
page_id: 22694
display_title: "Nextflow"
---

`<languages />`{=html} [Nextflow](https://www.nextflow.io) is software for running reproducible scientific workflows. The term `<i>`{=html}Nextflow`</i>`{=html} is used to describe both the domain-specific-language (DSL) the pipelines are written in, and the software used to interpret those workflows.

## Usage

On our systems, Nextflow is provided as a module you can load with `module load nextflow`.

While you can build your own workflow, you can also rely on the published [nf-core](https://nf-co.re/) pipelines. We will describe here a simple configuration that will let you run nf-core pipelines on our systems and help you to configure Nextflow properly for your own pipelines.

Our example uses the `nf-core/smrnaseq` pipeline.

#### Installation

The following procedure is to be run on a login node.

Start by installing a pip package to help with the setup; please note that the nf-core tools can be slow to install.

``` bash
module purge # Make sure that previously loaded modules are not polluting the installation 
module load python/3.11
module load rust # New nf-core installations will err out if rust hasn't been loaded
module load postgresql # Will not use PostgresSQL here, but some Python modules which list psycopg2 as a dependency in the installation would crash without it.
python -m venv nf-core-env
source nf-core-env/bin/activate
python -m pip install nf_core==2.13
```

Set the name of the pipeline to be tested, and load Nextflow and [Apptainer](https://docs.alliancecan.ca/Apptainer "Apptainer"){.wikilink} (the successor to the [Singularity](https://docs.alliancecan.ca/Singularity "Singularity"){.wikilink} container utility). Nextflow integrates well with Apptainer.

``` bash
export NFCORE_PL=smrnaseq
export PL_VERSION=2.3.1
module load nextflow/23
module load apptainer/1
```

An important step is to download all the Apptainer images that will be used to run the pipeline at the same time we download the workflow itself. If this isn\'t done, Nextflow will try to download the images from the compute nodes, just before steps are executed. This would not work on most of our clusters since there is no internet connection on the compute nodes.

Create a folder where images will be stored and set the environment variable `NXF_SINGULARITY_CACHEDIR` to it. Workflow images tend to be big, so do not store them in your \$HOME space because it has a small quota. Instead, store them in `/project` space.

``` bash
mkdir /project/<def-group>/NXF_SINGULARITY_CACHEDIR
export NXF_SINGULARITY_CACHEDIR=/project/<def-group>/NXF_SINGULARITY_CACHEDIR
```

You should share this folder with other members of your group who are planning to use Nextflow with Apptainer, in order to reduce duplication. Also, you may add the `export` command to your `~/.bashrc` as a convenience.

The following command downloads the `smrnaseq` pipeline to your `/scratch` directory and puts all the images in the cache directory.

``` bash
cd ~/scratch
nf-core download   --container-cache-utilisation amend --container-system  singularity   --compress none -r ${PL_VERSION}  -p 6  ${NFCORE_PL}
# Alteratively, you can run the download tool in interactive mode
nf-core download
# Here is what your Singularity image cache will look like after completion:
ls  $NXF_SINGULARITY_CACHEDIR/
depot.galaxyproject.org-singularity-bioconvert-1.1.1--pyhdfd78af_0.img
depot.galaxyproject.org-singularity-blat-36--0.img
depot.galaxyproject.org-singularity-bowtie-1.3.1--py310h7b97f60_6.img
depot.galaxyproject.org-singularity-bowtie2-2.4.5--py39hd2f7db1_2.img
depot.galaxyproject.org-singularity-fastp-0.23.4--h5f740d0_0.img
depot.galaxyproject.org-singularity-fastqc-0.12.1--hdfd78af_0.img
depot.galaxyproject.org-singularity-fastx_toolkit-0.0.14--hdbdd923_11.img
depot.galaxyproject.org-singularity-mirdeep2-2.0.1.3--hdfd78af_1.img
depot.galaxyproject.org-singularity-mirtrace-1.0.1--hdfd78af_1.img
depot.galaxyproject.org-singularity-mulled-v2-0c13ef770dd7cc5c76c2ce23ba6669234cf03385-63be019f50581cc5dfe4fc0f73ae50f2d4d661f7-0.img
depot.galaxyproject.org-singularity-mulled-v2-419bd7f10b2b902489ac63bbaafc7db76f8e0ae1-f5ff7de321749bc7ae12f7e79a4b581497f4c8ce-0.img
depot.galaxyproject.org-singularity-mulled-v2-ffbf83a6b0ab6ec567a336cf349b80637135bca3-40128b496751b037e2bd85f6789e83d4ff8a4837-0.img
depot.galaxyproject.org-singularity-multiqc-1.21--pyhdfd78af_0.img
depot.galaxyproject.org-singularity-pigz-2.3.4.img
depot.galaxyproject.org-singularity-r-data.table-1.12.2.img
depot.galaxyproject.org-singularity-samtools-1.19.2--h50ea8bc_0.img
depot.galaxyproject.org-singularity-seqcluster-1.2.9--pyh5e36f6f_0.img
depot.galaxyproject.org-singularity-seqkit-2.6.1--h9ee0642_0.img
depot.galaxyproject.org-singularity-ubuntu-20.04.img
depot.galaxyproject.org-singularity-umicollapse-1.0.0--hdfd78af_1.img
depot.galaxyproject.org-singularity-umi_tools-1.1.5--py39hf95cd2a_0.img
quay.io-singularity-bioconvert-1.1.1--pyhdfd78af_0.img
quay.io-singularity-blat-36--0.img
quay.io-singularity-bowtie-1.3.1--py310h7b97f60_6.img
quay.io-singularity-bowtie2-2.4.5--py39hd2f7db1_2.img
quay.io-singularity-fastp-0.23.4--h5f740d0_0.img
quay.io-singularity-fastqc-0.12.1--hdfd78af_0.img
quay.io-singularity-fastx_toolkit-0.0.14--hdbdd923_11.img
quay.io-singularity-mirdeep2-2.0.1.3--hdfd78af_1.img
quay.io-singularity-mirtrace-1.0.1--hdfd78af_1.img
quay.io-singularity-mulled-v2-0c13ef770dd7cc5c76c2ce23ba6669234cf03385-63be019f50581cc5dfe4fc0f73ae50f2d4d661f7-0.img
quay.io-singularity-mulled-v2-419bd7f10b2b902489ac63bbaafc7db76f8e0ae1-f5ff7de321749bc7ae12f7e79a4b581497f4c8ce-0.img
quay.io-singularity-mulled-v2-ffbf83a6b0ab6ec567a336cf349b80637135bca3-40128b496751b037e2bd85f6789e83d4ff8a4837-0.img
quay.io-singularity-multiqc-1.21--pyhdfd78af_0.img
quay.io-singularity-pigz-2.3.4.img
quay.io-singularity-r-data.table-1.12.2.img
quay.io-singularity-samtools-1.19.2--h50ea8bc_0.img
quay.io-singularity-seqcluster-1.2.9--pyh5e36f6f_0.img
quay.io-singularity-seqkit-2.6.1--h9ee0642_0.img
quay.io-singularity-ubuntu-20.04.img
quay.io-singularity-umicollapse-1.0.0--hdfd78af_1.img
quay.io-singularity-umi_tools-1.1.5--py39hf95cd2a_0.img
singularity-bioconvert-1.1.1--pyhdfd78af_0.img
singularity-blat-36--0.img
singularity-bowtie-1.3.1--py310h7b97f60_6.img
singularity-bowtie2-2.4.5--py39hd2f7db1_2.img
singularity-fastp-0.23.4--h5f740d0_0.img
singularity-fastqc-0.12.1--hdfd78af_0.img
singularity-fastx_toolkit-0.0.14--hdbdd923_11.img
singularity-mirdeep2-2.0.1.3--hdfd78af_1.img
singularity-mirtrace-1.0.1--hdfd78af_1.img
singularity-mulled-v2-0c13ef770dd7cc5c76c2ce23ba6669234cf03385-63be019f50581cc5dfe4fc0f73ae50f2d4d661f7-0.img
singularity-mulled-v2-419bd7f10b2b902489ac63bbaafc7db76f8e0ae1-f5ff7de321749bc7ae12f7e79a4b581497f4c8ce-0.img
singularity-mulled-v2-ffbf83a6b0ab6ec567a336cf349b80637135bca3-40128b496751b037e2bd85f6789e83d4ff8a4837-0.img
singularity-multiqc-1.21--pyhdfd78af_0.img
singularity-pigz-2.3.4.img
singularity-r-data.table-1.12.2.img
singularity-samtools-1.19.2--h50ea8bc_0.img
singularity-seqcluster-1.2.9--pyh5e36f6f_0.img
singularity-seqkit-2.6.1--h9ee0642_0.img
singularity-ubuntu-20.04.img
singularity-umicollapse-1.0.0--hdfd78af_1.img
singularity-umi_tools-1.1.5--py39hf95cd2a_0.img
```

This workflow downloads 18 containers for a total of about 4Go and creates an `nf-core-${NFCORE_PL}_${PL_VERSION}` folder with the version number `X_X_X` and `config` subfolders. The `config` subfolder includes the [institutional configuration](https://github.com/nf-core/configs) while the workflow itself is in the other subfolder.

This is what a typical nf-core pipeline looks like:

``` bash
ls nf-core-${NFCORE_PL}_${PL_VERSION}/2_3_1
assets        CITATIONS.md        docs     modules          nextflow_schema.json  subworkflows
bin           CODE_OF_CONDUCT.md  LICENSE  modules.json     pyproject.toml        tower.yml
CHANGELOG.md  conf                main.nf  nextflow.config  README.md             workflows
```

When the pipeline is launched, Nextflow will look at the `nextflow.config` file in that folder and also at the `~/.nextflow/config` file (if it exists) in your home to control how to run the workflow. The nf-core pipelines all have a default configuration, a test configuration, and container configurations (singularity, podman, etc). You will need to provide a custom configuration for the cluster you are running on, a simple configuration is provided in next section. Nextflow pipelines could also run on [Trillium](https://docs.alliancecan.ca/Trillium "Trillium"){.wikilink} if they were designed with that specific cluster in mind, but we generally discourage you to run nf-core or any other generic Nextflow pipeline on Trillium.

#### A configuration for our clusters {#a_configuration_for_our_clusters}

Nf-core provides configurations for the following clusters: Narval, Rorqual, Trillium, Nibi, and Fir. The cluster is autodetected via the hostname with the exception of Fir which should be loaded using the `-profile` flag. This file can be placed in `~/.nextflow/config` [nf-core/configs GitHub repo](https://github.com/nf-core/configs/blob/master/conf/alliance_canada.config)

The `$SLURM_ACCOUNT` environment variable needs to be set, which looks like `def-pname`. For the sake of ease this can be set in `~.bashrc` The `singularity.autoMounts = true` bits ensure that all the cluster File Systems (`/project`, `/scratch`, `/home` & `/localscratch`) will be properly mounted inside the singularity container.

This configuration ensures that there are no more than 100 jobs in the Slurm queue and that only 60 jobs are submitted per minute. It indicates that Rorqual machines have 192 cores and 750 GB of RAM with a maximum walltime of one week (168 hours). These are different for Trillium, as Trillium has a different setup with nodes. CPU and memory cannot be specified as a whole node is requested at a time.

The config is linked to the system you are running on, but it is also related to the pipeline itself. For example, here cpu = 1 is the default value, but steps in the pipeline can have more than that. This can get quite complicated and labels in the `nf-core-smrnaseq_2.3.1/2_3_1/conf/base.config` file are used internally by the pipeline to identify a step with a non default configuration. We do not cover this more advanced topic here, but note that tweaking these labels could make a big difference in the queuing and execution time of your pipeline.

If the jobs fail with return codes 125 (`<i>`{=html}out of memory`</i>`{=html}) or 139 (`<i>`{=html}omm killed because the process used more memory than what was allowed by cgroup`</i>`{=html}), the lines `process.errorStrategy` and `process.memory` in the configuration make sure that they are automatically restarted with an additional 4GB of RAM.

#### Running the pipeline {#running_the_pipeline}

Use the two profiles provided by nf-core (`<i>`{=html}test`</i>`{=html} for nextflow\'s test dataset and `<i>`{=html}singularity`</i>`{=html} for the container platform) and the profile we have just created for Narval. Note that Nextflow is mainly written in Java which tends to use a lot of virtual memory. On some clusters, this may be a problem when running from a login node.

``` bash
nextflow run nf-core-${NFCORE_PL}_${PL_VERSION}/2_3_1/  -profile test,singularity,narval  --outdir ${NFCORE_PL}_OUTPUT
```

Be careful if you have an AWS configuration in your `~/.aws` directory, as Nextflow might complain that it can\'t dowload the pipeline test dataset with your default id.

So now you have started Nextflow on the login node. This process sends jobs to Slurm when they are ready to be processed.

You can see the progression of the pipeline. You can also open a new session on the cluster or detach from the tmux session to have a look at the jobs in the Slurm queue with `squeue -u $USER` or `sq`

## Known issues {#known_issues}

#### \"unable to create native thread\" {#unable_to_create_native_thread}

The following error has been observed:

    java.lang.OutOfMemoryError: unable to create native thread: possibly out of memory or process/resource limits reached
    [error][gc,task] GC Failed to create worker thread

We believe this is due to Java trying to create threads to match the number of physical cores on a machine. Setting `export NXF_OPTS='-XX:ActiveProcessorCount=1'` when executing `nextflow` is reported to solve the problem.

#### SIGBUS

Some users have reported getting a `SIGBUS` error from the Nextflow main process. We suspect this is connected with these Nextflow issues:

`* `[`https://github.com/nextflow-io/nextflow/issues/842`](https://github.com/nextflow-io/nextflow/issues/842)\
`* `[`https://github.com/nextflow-io/nextflow/issues/2774`](https://github.com/nextflow-io/nextflow/issues/2774)

Setting the environment variable `NXF_OPTS="-Dleveldb.mmap=false"` when executing `nextflow` is reported to solve the problem.
