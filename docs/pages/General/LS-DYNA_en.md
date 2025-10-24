---
title: "LS-DYNA/en"
url: "https://docs.alliancecan.ca/wiki/LS-DYNA/en"
category: "General"
last_modified: "2025-10-21T14:19:23Z"
page_id: 14902
display_title: "LS-DYNA"
---

`<languages />`{=html}

# Introduction

[LS-DYNA](http://www.lstc.com) is available on all our clusters. It is used for many [applications](http://www.lstc.com/applications) to solve problems in multiphysics, solid mechanics, heat transfer and fluid dynamics. Analyses are performed as separate phenomena or coupled physics simulations such as thermal stress or fluid structure interaction. LSTC was recently purchased by Ansys, so the LS-DYNA software may eventually be exclusively provided as part of the Ansys module. For now, we recommend using the LS-DYNA software traditionally provided by LSTC as documented in this wiki page.

# Licensing

The Alliance is a hosting provider for LS-DYNA. This means that we have LS-DYNA software installed on our clusters. The Alliance does NOT however provide a generic license accessible to everyone or provide license hosting services. Instead, many institutions, faculties, and departments already have licenses that can be used on our clusters. So that such licenses can be reached from a cluster\'s compute nodes, some cluster-specific network changes will generally need to be done. In cases where a license has already been used on a particular cluster, these changes may already be done. Users unable to locate or arrange for a license on campus may contact [CMC Microsystems](https://www.cmc.ca/support/). Licenses purchased from CMC do not require the overhead of hosting a local license server since they are hosted on a remote server system that CMC manages with the added benefit of being usable anywhere. If you have your own server and need a quote for a locally managed license, consider contacting [Simutech](https://simutechgroup.com) or contact Ansys directly. SHARCNET does not provide any free LS-DYNA licenses or license hosting services at this time.

### Initial setup and testing {#initial_setup_and_testing}

If your (existing or new) license server has never been used on the cluster where you plan to run jobs, firewall changes will first need to be done on both the cluster side and server side. This will typically require involvement from both our technical team and the technical people managing your license software. To arrange this, send an email containing the service port and IP address of your floating license server to [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink}. To check if your license file is working run the following commands

`<code>`{=html}`module load ls-dyna`\
`ls-dyna_s or ls-dyna_d``</code>`{=html}

You don\'t need to specify any input file or arguments to run this test. The output header should contain a (non-empty) value for `Licensed to:` with the exception of CMC license servers. Press \^C to quit the program and return to the command line.

## Configuring your license {#configuring_your_license}

In 2019 Ansys, purchased the Livermore Software Technology Corporation (LSTC), developer of LS-DYNA. LS-DYNA licenses issued by Ansys since that time use `<b>`{=html}Ansys license servers`</b>`{=html}. Licenses issued by LSTC may still use an `<b>`{=html}LSTC license server`</b>`{=html}. You can also obtain an LS-DYNA license through [CMC Microsystems](https://www.cmc.ca/). This section explains how to configure your account or job script for each of these cases.

### LSTC license {#lstc_license}

If you have a license issued to run on a LSTC license server, there are two options to specify it:

Option 1) Specify your license server by creating a small file named `ls-dyna.lic` with the following contents:

where `<port>`{=html} is an integer number and `<server>`{=html} is the hostname of your LSTC license server. Put this file in directory `$HOME/.licenses/` on each cluster where you plan to submit jobs. The values in the file are picked up by LS-DYNA when it runs. This occurs because our module system sets the LSTC_FILE variable to `LSTC_FILE=/home/$USER.licenses/ls-dyna.lic` whenever you load a `ls-dyna` or `ls-dyna-mpi` module. This approach is recommended for users with a license hosted on a LSTC license server since (compared to the next option) the identical settings will automatically be used by all jobs you submit on the cluster (without the need to specify them in each individual slurm script or setting them in your environment).

Option 2) Specify your license server by setting the following two environment variables in your slurm scripts:

`export LSTC_LICENSE=network`\
`export LSTC_LICENSE_SERVER=``<port>`{=html}`@``<server>`{=html}

where `<port>`{=html} is an integer number and `<server>`{=html} is the hostname or IP address of your LSTC license server. These variables will take priority over any values specified in your `~/.licenses/ls-dyna.lic` file which must exist (even if it\'s empty) for any `ls-dyna` or `ls-dyna-mpi` module to successfully load. To ensure it exists, run `touch ~/.licenses/ls-dyna.lic` once on the command line on each cluster where you will submit jobs. For further details, see the official [documentation](https://lsdyna.ansys.com/download-install-overview/).

### Ansys license {#ansys_license}

If your LS-DYNA license is hosted on an Ansys license server, set the following two environment variables in your slurm scripts:

`export LSTC_LICENSE=ansys`\
`export ANSYSLMD_LICENSE_FILE=``<port>`{=html}`@``<server>`{=html}

where `<port>`{=html} is an integer number and `<server>`{=html} is the hostname or IP address of your Ansys license server. These variables cannot be defined in your `~/.licenses/ls-dyna.lic` file. The file however must exist (even if it\'s empty) for any `ls-dyna` module to load. To ensure this, run `touch ~/.licenses/ls-dyna.lic` once from the command line (or each time in your slurm scripts). Note that only module versions \>= 12.2.1 will work with Ansys license servers.

#### SHARCNET

The SHARCNET Ansys license supports running SMP and MPP LS-DYNA jobs. It can be used for free by anyone (on a core and job limited basis) on Nibi cluster by adding the following lines to your slurm script:

`export LSTC_LICENSE=ansys`\
`export ANSYSLMD_LICENSE_FILE=1055@license1.computecanada.ca`

### CMC license {#cmc_license}

If your LS-DYNA license was purchased from CMC, set the following two environment variables according to the cluster you are using:

`export LSTC_LICENSE=ansys`\
`Fir:      export ANSYSLMD_LICENSE_FILE=6624@172.26.0.101`\
`Nibi:     export ANSYSLMD_LICENSE_FILE=6624@10.25.1.56`\
`Narval:   export ANSYSLMD_LICENSE_FILE=6624@10.100.64.10`\
`Rorqual:  export ANSYSLMD_LICENSE_FILE=6624@10.100.64.10`\
`Trillium: export ANSYSLMD_LICENSE_FILE=6624@172.16.205.198`

where the IP address corresponds to the respective CADpass servers. No firewall changes are required to use a CMC license on any cluster since these have already been done. Since the remote CMC server that hosts LS=DYNA licenses is Ansys-based, these variables cannot be defined in your `~/.licenses/ls-dyna.lic` file. The file however must exist (even if it\'s empty) for any `ls-dyna` module to load. To ensure this is the case, run `touch ~/.licenses/ls-dyna.lic` once from the command line (or each time in your slurm scripts). Note that only module versions \>= 13.1.1 will work with Ansys license servers.

# Cluster job submission {#cluster_job_submission}

LS-DYNA provides binaries for running jobs on a single compute node (SMP - Shared Memory Parallel using OpenMP) or across multiple compute nodes (MPP - Message Passing Parallel using MPI). This section provides slurm scripts for each job type.

## Single node jobs {#single_node_jobs}

Modules for running jobs on a single compute node can be listed with: `module spider ls-dyna`. Jobs may be submitted to the queue with: `sbatch script-smp.sh`. The following slurm script shows how to run LS-DYNA with 8 cores on a single compute node. Regarding the AUTO option of the LSTC_MEMORY [environment variable](https://www.dynasupport.com/howtos/general/environment-variables), this setting allows memory to be dynamically extended beyond the specified `memory=1500M` word setting where it is suitable for explicit analysis such as metal forming simulations but not crash analysis. Given there are 4 Bytes/word for the single precision solver and 8 Bytes/word for the double precision solver, the 1500M setting in the slurm script example below equates to either 1) a maximum amount of (1500Mw\*8Bytes/w) = 12GB memory before LS-DYNA self-terminates when solving an implicit problem or 2) a starting amount of 12GB memory prior to extending it (up 25% if necessary) when solving an explicit problem assuming `LSTC_MEMORY=AUTO` is uncommented. Note that 12GB represents 75% of the total mem=16GB reserved for the job and is considered ideal for implicit jobs on a single node. To summarize, for both implicit and explicit analysis, once an estimate for the total solver memory is determined in GB, the total memory setting for slurm can be determined by multiplying by 25% while the memory parameter value in mega words can be calculated as (0.75\*memGB/8Bytes/w)\*1000M and (0.75\*memGB/4Bytes/w)\*1000M for double and single precision solutions respectively.

where

- ls-dyna_s = single precision smp solver
- ls-dyna_d = double precision smp solver

## Multiple node jobs {#multiple_node_jobs}

There are several modules installed for running jobs on multiple nodes using the MPP (Message Passing Parallel) version of LS-DYNA. The method is based on mpi and can scale to very many cores (8 or more). The modules may be listed by running `module spider ls-dyna-mpi`. Sample slurm scripts below demonstrate how to use these modules for submitting jobs to a specified number of whole nodes \*OR\* a specified total number of cores using `sbatch script-mpp-bynode.sh` or `sbatch script-mpp-bycore.sh` respectively. The MPP version requires a sufficiently large enough amount of memory (memory1) for the first core (processor 0) on the master node to decompose and simulate the model. This amount may be satisfied by specifying a value of `mem-per-cpu` to slurm slightly larger than the memory (memory2) required per core for simulation and then placing enough cores on the master node such that their differential sum (mem-per-cpu less memory2) is greater than or equal to memory1. Similar to the single node model, for best results, keep the sum of all expected memory per node within 75% of the reserved ram on a node. Thus in the first script below, assuming a 128GB full node memory compute node, memory1 maybe 6000M (48GB) maximum and memory2 200M (48GB/31cores).

### Specify node count {#specify_node_count}

Jobs can be submitted to a specified number of `<b>`{=html}whole`</b>`{=html} compute nodes with the following script.

where

- ls-dyna_s = single precision mpp solver
- ls-dyna_d = double precision mpp solver

### Specify core count {#specify_core_count}

Jobs can be submitted to an arbitrary number of compute nodes by specifying the number of cores. This approach allows the scheduler to determine the optimal number of compute nodes to minimize job wait time in the queue. Memory limits are applied per core, therefore a sufficiently large value of `mem-per-cpu` must be specified so the master processor can successfully decompose and handle its computations as explained in more detail in the opening paragraph of this section.

where

- ls-dyna_s = single precision mpp solver
- ls-dyna_d = double precision mpp solver

## Performance testing {#performance_testing}

Depending on the simulation LS-DYNA may not be able to efficiently use very many cores in parallel. Scaling test jobs should therefore always be run before submitting long jobs. Doing this will help determine the maximum number of cores that can be used before performance degradation begins to occur. To extract test job statistics such as Job Wall-clock time, CPU Efficiency and Memory Efficiency either the `seff jobnumber` command or a cluster job portal such as [this](https://portal.nibi.sharcnet.ca) can be used. In the past scaling test jobs for the standard airbag problem have shown significantly different performance characteristics in the past depending which cluster they were being run on. These tests however were rather small using only 6 cores on a single node with the ls-dyna/12.2.1 module and 6 cores evenly distributed across two nodes with the ls-dyna-mpi/12.2.1 module. Scaling tests should instead be run using the actual research simulation and cluster where the full production runs will be done to get reliable results.

# Graphical use {#graphical_use}

LSTC provides [LS-PrePost](https://www.lstc.com/products/ls-prepost) for pre- and post-processing of LS-DYNA [models](https://www.dynaexamples.com/). This program is made available by a separate module and does not require a license. To run abaqus graphically in a remote gui desktop do one of the following where the OnDemand desktop approach is recommended :

## OnDemand

1\. Connect to an OnDemand system using one of the following URLs in your laptop browser :\
[NIBI](https://docs.alliancecan.ca/wiki/Nibi#Access_through_Open_OnDemand_(OOD)): [`https://ondemand.sharcnet.ca`](https://ondemand.sharcnet.ca)

`FIR: `[`https://jupyterhub.fir.alliancecan.ca`](https://jupyterhub.fir.alliancecan.ca)\
`RORQUAL: `[`https://jupyterhub.rorqual.alliancecan.ca`](https://jupyterhub.rorqual.alliancecan.ca)\
`TRILLIUM: `[`https://ondemand.scinet.utoronto.ca`](https://ondemand.scinet.utoronto.ca)

2\. Open a new terminal window in your desktop and run :

`module load StdEnv/2020`\
`module load ls-prepost/4.9`\
`lsprepost OR lspp49`

## VncViewer

1\. Connect with a VncViewer client to a login or compute node by following [TigerVNC](https://docs.alliancecan.ca/VNC "TigerVNC"){.wikilink}\
2. Open a new terminal window in your desktop and run:

`module load StdEnv/2020`\
`module load ls-prepost/4.9`\
`lsprepost OR lspp49`
