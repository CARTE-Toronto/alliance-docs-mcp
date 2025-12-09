---
title: "COMSOL"
url: "https://docs.alliancecan.ca/wiki/COMSOL"
category: "General"
last_modified: "2025-12-06T15:16:28Z"
page_id: 5690
display_title: "COMSOL"
---

`<languages />`{=html}

`<translate>`{=html}

# Introduction

[COMSOL](http://www.comsol.com) is a general-purpose software for modelling engineering applications. We would like to thank COMSOL, Inc. for allowing its software to be hosted on our clusters via a special agreement. ![](Logo_comsol_blue_1571x143.png "Logo_comsol_blue_1571x143.png") We recommend that you consult the documentation included with the software under `<i>`{=html}File \> Help \> Documentation`</i>`{=html} prior to attempting to use COMSOL on one of our clusters. Links to the COMSOL blog, Knowledge Base, Support Centre and Documentation can be found at the bottom of the [COMSOL home page](http://www.comsol.com). Searchable online COMSOL documentation is also available [here](https://doc.comsol.com/).

# Licensing

We are a hosting provider for COMSOL. This means that we have COMSOL software installed on our clusters, but we do not provide a generic license accessible to everyone. Many institutions, faculties, and departments already have licenses that can be used on our clusters. Alternatively, you can purchase a license from [CMC](https://account.cmc.ca/en/WhatWeOffer/Products/CMC-00200-00368.aspx) for use anywhere in Canada. Once the legal aspects are worked out for licensing, there will be remaining technical aspects. The license server on your end will need to be reachable by our compute nodes. This will require our technical team to get in touch with the technical people managing your license software. If you have purchased a CMC license and will be connecting to the CMC license server, this has already been done. Once the license server work is done and your `<i>`{=html}\~/.licenses/comsol.lic`</i>`{=html} has been created, you can load any COMSOL module and begin using the software. If this is not the case, please contact our [technical support](https://docs.alliancecan.ca/technical_support "wikilink").

## Configuring your own license file {#configuring_your_own_license_file}

Our COMSOL module is designed to look for license information in a few places, one of which is your `<I>`{=html}\~/.licenses`</I>`{=html} directory. If you have your own license server then specify it by creating a text file `$HOME/.licenses/comsol.lic` with the following information:

Where `<server>`{=html} is your license server hostname and `<port>`{=html} is the flex port number of the license server.

### Local license setup {#local_license_setup}

For researchers wanting to use a new local institutional license server, firewall changes will need to be done to the network on both the Alliance (system/cluster) side and the institutional (server) side. To arrange this, send an email to [technical support](https://docs.alliancecan.ca/technical_support "wikilink") containing 1) the COMSOL lmgrd TCP flex port number (typically 1718 default) and 2) the static LMCOMSOL TCP vendor port number (typically 1719 default) and finally 3) the fully qualified hostname of your COMSOL license server. Once this is complete, create a corresponding `<i>`{=html}comsol.lic`</i>`{=html} text file as shown above.

### CMC license setup {#cmc_license_setup}

Researchers who own a COMSOL license subscription from CMC should use the following preconfigured public IP settings in their `<i>`{=html}comsol.lic`</i>`{=html} file:

-   Fir: SERVER 172.26.0.101 ANY 6601
-   Nibi: SERVER 10.25.1.56 ANY 6601
-   Narval/Rorqual: SERVER 10.100.64.10 ANY 6601
-   Trillium: SERVER nia-cmc ANY 6601

For example, a license file created on Nibi cluster would look as follows:

`[l2:~] cat ~/.licenses/comsol.lic`\
`SERVER 10.25.1.56 ANY 6601`\
`USE_SERVER`

If initial license checkout attempts fail, create a support case with [CMC](https://www.cmc.ca/support/)

## Installed products {#installed_products}

To check which [modules and products](https://www.comsol.com/products) are available for use, start COMSOL in [graphical mode](https://docs.alliancecan.ca/#Graphical_use "wikilink") and then click `<i>`{=html}Options -\> Licensed and Used Products`</i>`{=html} on the upper pull-down menu. For a more detailed explanation, click [here](https://doc.comsol.com/6.0/docserver/#!/com.comsol.help.comsol/comsol_ref_customizing.16.09.html). If a module/product is missing or reports being unlicensed, contact [technical support](https://docs.alliancecan.ca/technical_support "wikilink") as a reinstall of the CVMFS module you are using may be required.

## Installed versions {#installed_versions}

To check the full version number either start comsol in [gui](https://docs.alliancecan.ca/wiki/COMSOL#Graphical_use) mode and inspect the lower right corner messages window OR more simply login to a cluster and run comsol in batch mode as follows:

`[login-node:~] salloc --time=0:01:00 --nodes=1 --cores=1 --mem=1G --account=def-someuser`\
`[login-node:~] module load comsol/6.2`\
`[login-node:~] comsol batch -version`\
`COMSOL Multiphysics 6.2.0.290`

which corresponds to COMSOL 6.2 Update 1. In other words, when a new [COMSOL release](https://www.comsol.com/release-history) is installed, it will use the abbreviated 6.X version format but for convenience will contain the latest available update at the time of installation. As additional [product updates](https://www.comsol.com/product-update) are released they will instead utilize the full 6.X.Y.Z version format. For example, [Update 3](https://www.comsol.com/product-update/6.2) can be loaded on a cluster with the `module load comsol/6.2.0.415` OR `module load comsol` commands. We recommend using the moat recent update to take advantage of all the latest improvements. That said, if you want to continue using any module version (6.X or 6.X.Y.Z). you can be assured by definition that the software contained in these modules will remain exactly the same.

To check which versions are available in the standard environment you have loaded ( typically `StdEnv/2023` ) run the `module avail comsol` command. Lastly, to check which versions are available in ALL available standard environments, use the `module spider comsol` command.

A module `comsol/6.3` corresponding to version [6.3.0.290](https://www.comsol.com/product-download/6.3) is now available on all clusters.

# Submit jobs {#submit_jobs}

## Single compute node {#single_compute_node}

Sample submission script to run a COMSOL job with eight cores on a single compute node: `{{File
|name=mysub1.sh
|lang="bash"
|contents=
#!/bin/bash
#SBATCH --time=0-03:00             # Specify (d-hh:mm)
#SBATCH --account=def-group        # Specify (some account)
#SBATCH --mem=32G                  # Specify (set to 0 to use all memory on each node)
#SBATCH --cpus-per-task=8          # Specify (set to 32or44 graham, 32or48 cedar, 40 beluga, 48or64 narval to use all cores)
#SBATCH --nodes=1                  # Do not change
#SBATCH --ntasks-per-node=1        # Do not change

<!--T:205-->
INPUTFILE="ModelToSolve.mph"       # Specify input filename
OUTPUTFILE="SolvedModel.mph"       # Specify output filename

<!--T:206-->
# module load StdEnv/2020          # Versions < 6.2
module load StdEnv/2023
module load comsol/6.2

<!--T:10-->
comsol batch -inputfile ${INPUTFILE} -outputfile ${OUTPUTFILE} -np $SLURM_CPUS_ON_NODE
}}`{=mediawiki}

Depending on the complexity of the simulation, COMSOL may not be able to efficiently use very many cores. Therefore, it is advisable to test the scaling of your simulation by gradually increasing the number of cores. If near-linear speedup is obtained using all cores on a compute node, consider running the job over multiple full nodes using the next Slurm script.

## Multiple compute nodes {#multiple_compute_nodes}

Sample submission script to run a COMSOL job with eight cores distributed evenly over two compute nodes. Ideal for very large simulations (that exceed the capabilities of a single compute node), this script supports restarting interrupted jobs, allocating large temporary files to /scratch and utilizing the default `<i>`{=html}comsolbatch.ini`</i>`{=html} file settings. There is also an option to modify the Java heap memory described below the script.

```{=mediawiki}
{{File
|name=script-dis.sh
|lang="bash"
|contents=
#!/bin/bash
#SBATCH --time=0-03:00             # Specify (d-hh:mm)
#SBATCH --account=def-account      # Specify (some account)
#SBATCH --mem=16G                  # Specify (set to 0 to use all memory on each node)
#SBATCH --cpus-per-task=4          # Specify (set to 32or44 graham, 32or48 cedar, 40 beluga, 48or64 narval to use all cores)
#SBATCH --nodes=2                  # Specify (the number of compute nodes to use for the job)
#SBATCH --ntasks-per-node=1        # Do not change

<!--T:207-->
INPUTFILE="ModelToSolve.mph"       # Specify input filename
OUTPUTFILE="SolvedModel.mph"       # Specify output filename

<!--T:208-->
# module load StdEnv/2020          # Versions < 6.2
module load StdEnv/2023
module load comsol/6.2

<!--T:209-->
RECOVERYDIR=$SCRATCH/comsol/recoverydir
mkdir -p $RECOVERYDIR

<!--T:210-->
cp -f ${EBROOTCOMSOL}/bin/glnxa64/comsolbatch.ini comsolbatch.ini
cp -f ${EBROOTCOMSOL}/mli/startup/java.opts java.opts

<!--T:217-->
# export I_MPI_COLL_EXTERNAL=0      # Uncomment this line on narval 

<!--T:211-->
comsol batch -inputfile $INPUTFILE -outputfile $OUTPUTFILE -np $SLURM_CPUS_ON_NODE -nn $SLURM_NNODES \
-recoverydir $RECOVERYDIR -tmpdir $SLURM_TMPDIR -comsolinifile comsolbatch.ini -alivetime 15 \
# -recover -continue                # Uncomment this line to restart solving from latest recovery files

<!--T:221-->
}}
```
Note 1: If your multiple node job crashes on startup with a java segmentation fault, try increasing the java heap by adding the following two `sed` lines after the two `cp -f` lines. If it does not help, try further changing both 4g values to 8g. For further information see [Out of Memory](https://www.comsol.ch/support/knowledgebase/1243).

`sed -i 's/-Xmx2g/-Xmx4g/g' comsolbatch.ini`\
`sed -i 's/-Xmx768m/-Xmx4g/g' java.opts`

Note 2: On Narval, jobs may run slow when submitted with comsol/6.0.0.405 to multiple nodes using the above Slurm script. If this occurs, use comsol/6.0 instead and open a ticket to report the problem. The latest comsol/6.1.X modules have not been tested on Narval yet.

Note 3: On Graham, there is a small chance jobs will run slow or hang during startup when submitted to a single node with the above `<i>`{=html}script-smp.sh`</i>`{=html} script. If this occurs, use the multiple node `<i>`{=html}script-dis.sh`</i>`{=html} script instead adding `#SBATCH --nodes=1` and then open a ticket to report the problem.

# Graphical use {#graphical_use}

COMSOL can be run interactively in full graphical mode using either of the following methods.

## On cluster nodes {#on_cluster_nodes}

Suitable to interactively run computationally intensive test jobs using ALL available cores and memory reserved by `salloc` on a single cluster compute node:

:   1\) Connect to a compute node (3-hour time limit) with [TigerVNC](https://docs.alliancecan.ca/VNC#Compute_nodes "wikilink").
:   2\) Open a terminal window in vncviewer and run:
    :; `export XDG_RUNTIME_DIR=${SLURM_TMPDIR}`
:   3\) Start COMSOL Multiphysics 6.2 (or newer versions).
    :; `module load StdEnv/2023`

    :; `module load comsol/6.3`

    :; `comsol`
:   4\) Start COMSOL Multiphysics 5.6 (or newer versions).
    :; `module load StdEnv/2020`

    :; `module load comsol/5.6`

    :; `comsol`
:   5\) Start COMSOL Multiphysics 5.5 (or older versions).
    :; `module load StdEnv/2016`

    :; `module load comsol/5.5`

    :; `comsol`

## On VDI nodes {#on_vdi_nodes}

Suitable interactive use on gra-vdi includes: running compute calculations with a maximum of 12 cores and 128GB memory, creating or modifying simulation input files, performing post-processing or data visualization tasks. If you need more cores or memory when working in graphical mode, use COMSOL on a cluster compute node (as shown above) where you can reserve up to all available cores and/or memory on the node.

:   1\) Connect to gra-vdi (no time limit) with [TigerVNC](https://docs.alliancecan.ca/VNC#Compute_nodes "wikilink").
:   2\) Open a terminal window in vncviewer.
:   3\) Start COMSOL Multiphysics 6.2 (or newer versions).
    :; `module load CcEnv StdEnv/2023`

    :; `module avail comsol`

    :; `module load comsol/6.3`

    :; `comsol -np 12 -Dosgi.locking=none`
:   4\) Start COMSOL Multiphysics 6.2 (or older versions).
    :; `module load CcEnv StdEnv/2020`

    :; `module avail comsol`

    :; `module load comsol/5.6`

    :; `comsol -np 12 -Dosgi.locking=none`
:   5\) Start COMSOL Multiphysics 5.5 (or older versions).
    :; `module load CcEnv StdEnv/2016`

    :; `module avail comsol`

    :; `module load comsol/5.5`

    :; `comsol -np 12 -Dosgi.locking=none`

Note: If all the upper menu items are greyed out immediately after COMSOL starts in GUI mode, and are therefore not clickable, your `<i>`{=html}\~/.comsol`</i>`{=html} maybe corrupted. To fix the problem rename or remove it by running `rm -rf ~/.comsol` and try starting COMSOL again. This may occur if you previously loaded a COMSOL module from the SnEnv on gra-vdi.

# Parameter sweeps {#parameter_sweeps}

## Batch sweep {#batch_sweep}

When working interactively in the COMSOL GUI, parametric problems may be solved using the [Batch Sweep](https://www.comsol.com/blogs/the-power-of-the-batch-sweep/) approach. Multiple parameter sweeps maybe carried out as shown in [this video](https://www.comsol.com/video/performing-parametric-sweep-study-comsol-multiphysics). Speedup due to [Task Parallism](https://www.comsol.com/blogs/added-value-task-parallelism-batch-sweeps/) may also be realized.

## Cluster sweep {#cluster_sweep}

To run a parameter sweep on a cluster, a job must be submitted to the scheduler from the command line using `sbatch slurmscript`. For a discussion regarding additional required arguments, see [a](https://www.comsol.com/support/knowledgebase/1250) and [b](https://www.comsol.com/blogs/how-to-use-job-sequences-to-save-data-after-solving-your-model/) for details. Support for submitting parametric simulations to the cluster queue from the graphical interface using a [Cluster Sweep node](https://www.comsol.com/blogs/how-to-use-the-cluster-sweep-node-in-comsol-multiphysics/) is not available at this time. `</translate>`{=html}
