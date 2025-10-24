---
title: "Ansys"
url: "https://docs.alliancecan.ca/wiki/Ansys"
category: "General"
last_modified: "2025-10-21T20:57:31Z"
page_id: 4568
display_title: "Ansys"
---

`<languages />`{=html}

`<translate>`{=html} [Ansys](http://www.ansys.com/) is a software suite for engineering simulation and 3-D design. It includes packages such as [Ansys Fluent](http://www.ansys.com/Products/Fluids/ANSYS-Fluent) and [Ansys CFX](http://www.ansys.com/products/fluids/ansys-cfx).

# Licensing

We are a hosting provider for Ansys. This means that we have the software installed on our clusters, but we do not provide a generic license accessible to everyone. However, many institutions, faculties, and departments already have licenses that can be used on our clusters. Once the legal aspects are worked out for licensing, there will be remaining technical aspects. The license server on your end will need to be reachable by our compute nodes. This will require our technical team to get in touch with the technical people managing your license software. In some cases, this has already been done. You should then be able to load the Ansys module, and it should find its license automatically. If this is not the case, please contact our [technical support](https://docs.alliancecan.ca/technical_support "wikilink") so that they can arrange this for you.

## Configuring your license file {#configuring_your_license_file}

Our module for Ansys is designed to look for license information in a few places. One of those places is your /home folder. You can specify your license server by creating a file named `$HOME/.licenses/ansys.lic` consisting of two lines as shown. Customize the file by replacing FLEXPORT and LICSERVER with the appropriate values for your server.

  -----------------------------------------------------------------------------------------------------
  setenv(\"ANSYSLMD_LICENSE_FILE\", \"`<b>`{=html}FLEXPORT`</span>`{=html}\@LICSERVER`</b>`{=html}\")
  -----------------------------------------------------------------------------------------------------

  : style=\"text-align:left; background-color:#F2F2F2; font-size:110%\" \| FILE: ansys.lic

The following table provides established values for the CMC and SHARCNET license servers. To use a different server, locate the corresponding values as explained in [Local license servers](https://docs.alliancecan.ca/#Local_license_servers "wikilink").

  License    System/Cluster   LICSERVER                     FLEXPORT   NOTES
  ---------- ---------------- ----------------------------- ---------- ------------------------------------------------------------------------
  CMC        fir              `172.26.0.101`                `6624`     None
  CMC        narval/rorqual   `10.100.64.10`                `6624`     None
  CMC        nibi             `10.25.1.56`                  `6624`     NewIP Feb21/2025
  CMC        trillium         `172.16.205.198`              `6624`     None
  SHARCNET   fir              `license1.computecanada.ca`   `1055`     Currently `<span style="Color:red">`{=html}NOT`</span>`{=html} working
  SHARCNET   narval/rorqual   `license3.sharcnet.ca`        `1055`     Supports \<= ansys/2024R2.04
  SHARCNET   nibi             `license1.computecanada.ca`   `1055`     Supports \<= ansys/2025R1.02
  SHARCNET   trillium         `localhost`                   `1055`     Currently `<span style="Color:red">`{=html}NOT`</span>`{=html} working

  : style=\"text-align:left; background-color:#F2F2F2; font-size:110%\" \| TABLE: Preconfigured license servers

Researchers who purchase a new CMC license subscription must [submit](https://www.cmc.ca/support/) your Alliance account username otherwise license checkouts will fail. The number of cores that can be used with a CMC license is described in the `<i>`{=html}Other Tricks and Tips`</i>`{=html} sections of the [Ansys Electronics Desktop and Ansys Mechanical/Fluids quick start guides](https://www.cmc.ca/?s=Other+Tricks+and+Tips&lang=en/).

### Local license servers {#local_license_servers}

Before a local institutional Ansys license server can be used on the Alliance, firewall changes will need to be done on both the server and cluster side. For many Ansys servers this work has already been done and they can be used by following the steps in the \"Ready To Use\" section below. For Ansys servers that have never used on the Alliance, two additional steps must be done as shown in the \"Setup Required\" section also below.

#### Ready to use {#ready_to_use}

To use a local institutional ANSYS License server whose network/firewall connections have already been setup for use on an Alliance cluster, contact your local Ansys license server administrator and get the following two pieces of information:

`1) the configured Ansys flex port (FLEXPORT) number commonly 1055`\
`2) the fully qualified hostname (LICSERVER) of the license server`

Then simply configure your `~/.licenses/ansys.lic` file by plugging the values for FLEXPORT and LICSERVER into the `FILE: ansys.lic` template above.

#### Setup required {#setup_required}

To use a local Ansys license server that has never been setup for use with an Alliance cluster, then you will ALSO need to get the following from your local Ansys license server administrator:

` 3) the statically configured vendor port (VENDPORT) of the license server`\
` 4) confirmation ``<servername>`{=html}` will resolve to the same IP address as LICSERVER on the cluster`

where the `<servername>`{=html} can be found in the first line of the license file with format \"SERVER `<servername>`{=html} `<host id>`{=html} `<lmgrd port>`{=html}\". Send items 1-\>3 by email to [technical support](https://docs.alliancecan.ca/technical_support "wikilink") and mention which Alliance cluster you want to run Ansys jobs on. An Alliance system administrator will then open the outbound cluster firewall so license checkout requests can reach your license server from the cluster compute nodes. A range of IP addresses will then be sent back to you. Give these to your local network administrator. Request the firewall into your local license server be opened so that Ansys license connection (checkout requests) can reach your servers FLEXPORT and VENDPORT ports across the IP range.

## Checking license {#checking_license}

To test if your `ansys.lic` is configured and working properly copy/paste the following sequence of commands on the cluster you are submitting jobs to. The only required change would be to specify YOURUSERID. If the software on a remote license server has not been updated then a failure can occur if the latest module version of ansys is loaded to test with. Therefore to be certain the license checkouts will work when jobs are run in the queue, the same ansys module version that you load in your slurm scripts should be specified below.

`[login-node:~] cd /tmp`\
`[login-node:~] salloc --time1:0:0 --mem1000M --accountdef-YOURUSERID`\
`[login-node:~] module load StdEnv/2023; module load ansys/2023R2`\
`[login-node:~] $EBROOTANSYS/v$(echo ${EBVERSIONANSYS:2:2}${EBVERSIONANSYS:5:1})/licensingclient/linx64/lmutil lmstat -c $ANSYSLMD_LICENSE_FILE | grep "ansyslmd: UP" 1> /dev/null && echo Success  echo Fail`

If `Success` is output license checkouts should work when jobs are submitted to the queue.\
If `Fail` is output then jobs will likely fail requiring a problem ticket to be submitted to resolve.\

# Version compatibility {#version_compatibility}

Ansys simulations are typically forward compatible but `<span style="color:red">`{=html}NOT`</span>`{=html} backwards compatible. This means that simulations created using an older version of Ansys can be expected to load and run fine with any newer version. For example, a simulation created and saved with ansys/2022R2 should load and run smoothly with ansys/2023R2 but `<span style="color:red">`{=html}NOT`</span>`{=html} the other way around. While it may be possible to start a simulation running with an older version random error messages or crashing will likely occur. Regarding Fluent simulations, if you cannot recall which version of ansys was used to create your cas file try grepping it as follows to look for clues :

\$ grep -ia fluent combustor.cas

`  (0 "fluent15.0.7  build-id: 596")`

\$ grep -ia fluent cavity.cas.h5

`  ANSYS_FLUENT 24.1 Build 1018`

## Platform support {#platform_support}

Ansys provides detailed platform support information describing software/hardware compatibility for the [Current Release](https://www.ansys.com/it-solutions/platform-support) and [Previous Releases](https://www.ansys.com/it-solutions/platform-support/previous-releases). The `<I>`{=html}Platform Support by Application / Product`</I>`{=html} pdf is of special interest since it shows which packages are supported under Windows but not under Linux and thus not on the Alliance such as Spaceclaim.

## What\'s new {#whats_new}

Ansys posts [Product Release and Updates](https://www.ansys.com/products/release-highlights) for the latest releases. Similar information for previous releases can generally be pulled up for various application topics by visiting the Ansys [blog](https://www.ansys.com/blog) page and using the FILTERS search bar. For example, searching on `What’s New Fluent 2024 gpu` pulls up a document with title [`What’s New for Ansys Fluent in 2024 R1?`](https://www.ansys.com/blog/fluent-2024-r1) containing a wealth of the latest gpu support information. Specifying a version number in the [Press Release](https://www.ansys.com/news-center/press-releases) search bar is also a good way to find new release information. Recently a module for the latest ANSYS release was installed `ansys/2025R1.02` however to use it requires a suitably updated license server such as CMCs. The upgrade of the SHARCNET license server is underway however until it is complete (and this message updated accordingly) it will only support jobs run with `ansys/2024R2.04` or older. To request a new version be installed or a problem with an exiting module please [submit a ticket](https://docs.alliancecan.ca/Technical_support "wikilink").

## Service packs {#service_packs}

Starting with Ansys 2024 a separate Ansys module will appear on the clusters with a decimal and two digits appearing after the release number whenever a service pack is been installed over the initial release. For example, the initial 2024 release with no service pack applied may be loaded with `module load ansys/2024R1` while a module with Service Pack 3 applied may be loaded with `module load ansys/2024R1.03` instead. If a service pack is already available by the time a new release is to be installed, then most likely only a module for that service pack number will be installed unless a request to install the initial release is also received.

Most users will likely want to load the latest module version equipped with the latest installed service pack which can be achieved by simply doing `module load ansys`. While it\'s not expected service packs will impact numerical results, the changes they make are extensive and so, if computations have already been done with the initial release or an earlier service pack then some groups may prefer to continue using it. Having separate modules for each service pack makes this possible. Starting with Ansys 2024R1 a detailed description of what each service pack does can be found by searching this [link](https://storage.ansys.com/staticfiles/cp/Readme/release2024R1/info_combined.pdf) for `<I>`{=html}Service Pack Details`</I>`{=html}. Future versions will presumably be similarly searchable by manually modifying the version number contained in the link.

# Cluster batch job submission {#cluster_batch_job_submission}

The Ansys software suite comes with multiple implementations of MPI to support parallel computation. Unfortunately, none of them support our [Slurm scheduler](https://docs.alliancecan.ca/Running_jobs "wikilink"). For this reason, we need special instructions for each Ansys package on how to start a parallel job. In the sections below, we give examples of submission scripts for some of the packages. While the slurm scripts should work with on all clusters, Niagara users may need to make some additional changes covered [here](https://docs.scinet.utoronto.ca/index.php).

## Ansys Fluent {#ansys_fluent}

Typically, you would use the following procedure to run Fluent on one of our clusters:

1.  Prepare your Fluent job using Fluent from the Ansys Workbench on your desktop machine up to the point where you would run the calculation.
2.  Export the \"case\" file with `<i>`{=html}File \> Export \> Case...`</i>`{=html} or find the folder where Fluent saves your project\'s files. The case file will often have a name like `FFF-1.cas.gz`.
3.  If you already have data from a previous calculation, which you want to continue, export a \"data\" file as well (`<i>`{=html}File \> Export \> Data...`</i>`{=html}) or find it in the same project folder (`FFF-1.dat.gz`).
4.  [Transfer](https://docs.alliancecan.ca/Transferring_data "wikilink") the case file (and if needed the data file) to a directory on the [/project](https://docs.alliancecan.ca/Project_layout "wikilink") or [/scratch](https://docs.alliancecan.ca/Storage_and_file_management#Storage_types "wikilink") filesystem on the cluster. When exporting, you can save the file(s) under a more instructive name than `FFF-1.*` or rename them when they are uploaded.
5.  Now you need to create a \"journal\" file. Its purpose is to load the case file (and optionally the data file), run the solver and finally write the results. See examples below and remember to adjust the filenames and desired number of iterations.
6.  If jobs frequently fail to start due to license shortages and manual resubmission of failed jobs is not convenient, consider modifying your script to requeue your job (up to 4 times) as shown under the `<i>`{=html}by node + requeue`</i>`{=html} tab further below. Be aware that doing this will also requeue simulations that fail due to non-license related issues (such as divergence), resulting in lost compute time. Therefore it is strongly recommended to monitor and inspect each Slurm output file to confirm each requeue attempt is license related. When it is determined that a job is requeued due to a simulation issue, immediately manually kill the job progression with `scancel jobid` and correct the problem.
7.  After [running the job](https://docs.alliancecan.ca/Running_jobs "wikilink"), you can download the data file and import it back into Fluent with `<i>`{=html}File \> Import \> Data...`</i>`{=html}.

### Slurm scripts {#slurm_scripts}

#### General purpose {#general_purpose}

Most Fluent jobs should use the following `<i>`{=html}by node`</i>`{=html} script to minimize solution latency and maximize performance over as few nodes as possible. Very large jobs, however, might wait less in the queue if they use a `<i>`{=html}by core`</i>`{=html} script. However, the startup time of a job using many nodes can be significantly longer, thus offsetting some of the benefits. In addition, be aware that running large jobs over an unspecified number of potentially very many nodes will make them far more vulnerable to crashing if any of the compute nodes fail during the simulation. The scripts will ensure Fluent uses shared memory for communication when run on a single node or distributed memory (utilizing MPI and the appropriate HPC interconnect) when run over multiple nodes. The two narval tabs may be be useful to provide a more robust alternative if fluent crashes during the initial auto mesh partitioning phase when using the standard intel based scripts with the parallel solver. The other option would be to manually perform the mesh partitioning in the fluent gui then try to run the job again on the cluster with the intel scripts. Doing so will allow you to inspect the partition statistics and specify the partitioning method to obtain an optimal result. The number of mesh partitions should be an integral multiple of the number of cores; for optimal efficiency, ensure at least 10000 cells per core.

`<tabs>`{=html}

`<tab name="Multinode (by node)">`{=html} `{{File
|name=script-flu-bynode-intel.sh
|lang="bash"
|contents=
#!/bin/bash

<!--T:2302-->
#SBATCH --account=def-group   # Specify account name
#SBATCH --time=00-03:00       # Specify time limit dd-hh:mm
#SBATCH --nodes=1             # Specify number of compute nodes (narval 1 node max)
#SBATCH --ntasks-per-node=32  # Specify number of cores per node (max all cores of a compute node)
#SBATCH --mem=0               # Do not change (allocates all memory per compute node)
#SBATCH --cpus-per-task=1     # Do not change

<!--T:2306-->
module load StdEnv/2023       # Do not change
module load ansys/2023R2      # or newer versions

<!--T:4733-->
MYJOURNALFILE=sample.jou      # Specify your journal file name
MYVERSION=3d                  # Specify 2d, 2ddp, 3d or 3ddp

<!--T:501-->
# ------- do not change any lines below --------

<!--T:6782-->
if [[ "${CC_CLUSTER}" == narval ]]; then
 if [ "$EBVERSIONGENTOO" == 2020 ]; then
   module load intel/2021 intelmpi
   export INTELMPI_ROOT=$I_MPI_ROOT/mpi/latest
   export HCOLL_RCACHE=^ucs
 elif [ "$EBVERSIONGENTOO" == 2023 ]; then
   module load intel/2023 intelmpi
   export INTELMPI_ROOT=$I_MPI_ROOT
 fi
 unset I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS
 unset I_MPI_ROOT
fi

<!--T:6783-->
slurm_hl2hl.py --format ANSYS-FLUENT > /tmp/machinefile-$SLURM_JOB_ID
NCORES=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE * SLURM_CPUS_PER_TASK))

<!--T:6784-->
if [ "$SLURM_NNODES" == 1 ]; then
 #export I_MPI_HYDRA_BOOTSTRAP=ssh    # uncomment on beluga or cedar
 fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pshmem -i $MYJOURNALFILE
else
 if [[ "${CC_CLUSTER}" == nibi ]]; then
   fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -peth -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
 else
   fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
 fi
fi
}}`{=mediawiki} `</tab>`{=html}

`<tab name="Multinode (by core)">`{=html} `{{File
|name=script-flu-bycore-intel.sh
|lang="bash"
|contents=
#!/bin/bash

<!--T:2202-->
#SBATCH --account=def-group   # Specify account
#SBATCH --time=00-03:00       # Specify time limit dd-hh:mm
##SBATCH --nodes=1            # Uncomment to specify (narval 1 node max)
#SBATCH --ntasks=16           # Specify total number of cores for all nodes
#SBATCH --mem-per-cpu=4G      # Specify memory per core
#SBATCH --cpus-per-task=1     # Do not change

<!--T:2206-->
module load StdEnv/2023       # Do not change
module load ansys/2023R2      # or newer versions

<!--T:4736-->
MYJOURNALFILE=sample.jou      # Specify your journal file name
MYVERSION=3d                  # Specify 2d, 2ddp, 3d or 3ddp

<!--T:502-->
# ------- do not change any lines below --------

<!--T:6785-->
if [[ "${CC_CLUSTER}" == narval ]]; then
 if [ "$EBVERSIONGENTOO" == 2020 ]; then
   module load intel/2021 intelmpi
   export INTELMPI_ROOT=$I_MPI_ROOT/mpi/latest
   export HCOLL_RCACHE=^ucs
 elif [ "$EBVERSIONGENTOO" == 2023 ]; then
   module load intel/2023 intelmpi
   export INTELMPI_ROOT=$I_MPI_ROOT
 fi
 unset I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS
 unset I_MPI_ROOT
fi

<!--T:6786-->
slurm_hl2hl.py --format ANSYS-FLUENT > /tmp/machinefile-$SLURM_JOB_ID
NCORES=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))

<!--T:6787-->
if [ "$SLURM_NNODES" == 1 ]; then
 #export I_MPI_HYDRA_BOOTSTRAP=ssh    # uncomment on beluga or cedar
 fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pshmem -i $MYJOURNALFILE
else
 if [[ "${CC_CLUSTER}" == nibi ]]; then
   fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -peth -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
 else
   fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
 fi
fi
}}`{=mediawiki} `</tab>`{=html}

`<tab name="Multinode (by node, narval)">`{=html}

`` uniq`; do echo "${i}:$(cat /tmp/mf-$SLURM_JOB_ID  grep $i  wc -l)" >> /tmp/machinefile-$SLURM_JOB_ID; done ``

NCORES=\$((SLURM_NNODES \* SLURM_NTASKS_PER_NODE \* SLURM_CPUS_PER_TASK))

if \[ \"\$SLURM_NNODES\" == 1 \]; then

`fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=openmpi -pshmem -i $MYJOURNALFILE`

else

`fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=openmpi -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE`

fi }} `</tab>`{=html}

`<tab name="Multinode (by core, narval)">`{=html}

`` uniq`; do echo "${i}:$(cat /tmp/mf-$SLURM_JOB_ID  grep $i  wc -l)" >> /tmp/machinefile-$SLURM_JOB_ID; done ``

NCORES=\$((SLURM_NTASKS \* SLURM_CPUS_PER_TASK))

if \[ \"\$SLURM_NNODES\" == 1 \]; then

`fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=openmpi -pshmem -i $MYJOURNALFILE`

else

`fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=openmpi -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE`

fi }} `</tab>`{=html}

`<tab name="Multinode (by node, trillium)">`{=html}

`</tab>`{=html}

`</tabs>`{=html}

#### License requeue {#license_requeue}

The scripts in this section should only be used with Fluent jobs that are known to complete normally without generating any errors in the output however typically require multiple requeue attempts to checkout licenses. They are not recommended for Fluent jobs that may 1) run for a long time before crashing 2) run to completion but contain unresolved journal file warnings, since in both cases the simulations will be repeated from the beginning until the maximum number of requeue attempts specified by the `array` value is reached. For these types of jobs, the general purpose scripts above should be used instead.

`<tabs>`{=html} `<tab name="Multinode (by node + requeue)">`{=html} `{{File
|name=script-flu-bynode+requeue.sh
|lang="bash"
|contents=
#!/bin/bash

<!--T:2402-->
#SBATCH --account=def-group   # Specify account
#SBATCH --time=00-03:00       # Specify time limit dd-hh:mm
#SBATCH --nodes=1             # Specify number of compute nodes (narval 1 node max)
#SBATCH --ntasks-per-node=32  # Specify number of cores per compute node
#SBATCH --mem=0               # Do not change (allocates all memory per compute node)
#SBATCH --cpus-per-task=1     # Do not change
#SBATCH --array=1-5%1         # Specify number of requeue attempts (2 or more, 5 is shown)

<!--T:2406-->
module load StdEnv/2023       # Do not change
module load ansys/2023R2      # Specify version (or newer)

<!--T:4739-->
MYJOURNALFILE=sample.jou      # Specify your journal file name
MYVERSION=3d                  # Specify 2d, 2ddp, 3d or 3ddp

<!--T:506-->
# ------- do not change any lines below --------

<!--T:4740-->
if [[ "${CC_CLUSTER}" == narval ]]; then
 if [ "$EBVERSIONGENTOO" == 2020 ]; then
   module load intel/2021 intelmpi
   export INTELMPI_ROOT=$I_MPI_ROOT/mpi/latest
   export HCOLL_RCACHE=^ucs
 elif [ "$EBVERSIONGENTOO" == 2023 ]; then
   module load intel/2023 intelmpi
   export INTELMPI_ROOT=$I_MPI_ROOT
 fi
 unset I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS
 unset I_MPI_ROOT
fi

<!--T:4741-->
slurm_hl2hl.py --format ANSYS-FLUENT > /tmp/machinefile-$SLURM_JOB_ID
NCORES=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE * SLURM_CPUS_PER_TASK))

<!--T:2410-->
if [ "$SLURM_NNODES" == 1 ]; then
 #export I_MPI_HYDRA_BOOTSTRAP=ssh
 fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pshmem -i $MYJOURNALFILE
else
 if [[ "${CC_CLUSTER}" == nibi ]]; then
   fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -peth -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
 else
   fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
 fi
fi
if [ $? -eq 0 ]; then
    echo "Job completed successfully! Exiting now."
    scancel $SLURM_ARRAY_JOB_ID
else
    echo "Job attempt $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT failed due to license or simulation issue!"
    if [ $SLURM_ARRAY_TASK_ID -lt $SLURM_ARRAY_TASK_COUNT ]; then
       echo "Resubmitting job now …"
    else
       echo "All job attempts failed exiting now."
    fi
fi
}}`{=mediawiki} `</tab>`{=html}

`<tab name="Multinode (by core + requeue)">`{=html} `{{File
|name=script-flu-bycore+requeue.sh
|lang="bash"
|contents=
#!/bin/bash

<!--T:2902-->
#SBATCH --account=def-group   # Specify account
#SBATCH --time=00-03:00       # Specify time limit dd-hh:mm
##SBATCH --nodes=1            # Uncomment to specify (narval 1 node max) 
#SBATCH --ntasks=16           # Specify total number of cores
#SBATCH --mem-per-cpu=4G      # Specify memory per core
#SBATCH --cpus-per-task=1     # Do not change
#SBATCH --array=1-5%1         # Specify number of requeue attempts (2 or more, 5 is shown)

<!--T:2906-->
module load StdEnv/2023       # Do not change
module load ansys/2023R2      # Specify version (or newer)

<!--T:4742-->
MYJOURNALFILE=sample.jou      # Specify your journal file name
MYVERSION=3d                  # Specify 2d, 2ddp, 3d or 3ddp

<!--T:507-->
# ------- do not change any lines below --------

<!--T:4743-->
if [[ "${CC_CLUSTER}" == narval ]]; then
 if [ "$EBVERSIONGENTOO" == 2020 ]; then
   module load intel/2021 intelmpi
   export INTELMPI_ROOT=$I_MPI_ROOT/mpi/latest
   export HCOLL_RCACHE=^ucs
 elif [ "$EBVERSIONGENTOO" == 2023 ]; then
   module load intel/2023 intelmpi
   export INTELMPI_ROOT=$I_MPI_ROOT
 fi
 unset I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS
 unset I_MPI_ROOT
fi

<!--T:4744-->
slurm_hl2hl.py --format ANSYS-FLUENT > /tmp/machinefile-$SLURM_JOB_ID
NCORES=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))

<!--T:2910-->
if [ "$SLURM_NNODES" == 1 ]; then
 #export I_MPI_HYDRA_BOOTSTRAP=ssh
 fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pshmem -i $MYJOURNALFILE
else
 if [[ "${CC_CLUSTER}" == nibi ]]; then
   fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -peth -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
 else
   fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
 fi
fi
if [ $? -eq 0 ]; then
    echo "Job completed successfully! Exiting now."
    scancel $SLURM_ARRAY_JOB_ID
else
    echo "Job attempt $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT failed due to license or simulation issue!"
    if [ $SLURM_ARRAY_TASK_ID -lt $SLURM_ARRAY_TASK_COUNT ]; then
       echo "Resubmitting job now …"
    else
       echo "All job attempts failed exiting now."
    fi
fi
}}`{=mediawiki} `</tab>`{=html} `</tabs>`{=html}

#### Solution restart {#solution_restart}

The following two scripts are provided to automate restarting very large jobs that require more than the typical seven-day maximum runtime window available on most clusters. Jobs are restarted from the most recent saved time step files. A fundamental requirement is the first time step can be completed within the requested job array time limit (specified at the top of your Slurm script) when starting a simulation from an initialized solution field. It is assumed that a standard fixed time step size is being used. To begin, a working set of sample.cas, sample.dat and sample.jou files must be present. Next edit your sample.jou file to contain `/solve/dual-time-iterate 1` and `/file/auto-save/data-frequency 1`. Then create a restart journal file by doing `cp sample.jou sample-restart.jou` and edit the sample-restart.jou file to contain `/file/read-cas-data sample-restart` instead of `/file/read-cas-data sample` and comment out the initialization line with a semicolon for instance `;/solve/initialize/initialize-flow`. If your 2nd and subsequent time steps are known to run twice as fast (as the initial time step), edit sample-restart.jou to specify `/solve/dual-time-iterate 2`. By doing this, the solution will only be restarted after two 2 time steps are completed following the initial time step. An output file for each time step will still be saved in the output subdirectory. The value 2 is arbitrary but should be chosen such that the time for 2 steps fits within the job array time limit. Doing this will minimize the number of solution restarts which are computationally expensive. If your first time step performed by sample.jou starts from a converged (previous) solution, choose 1 instead of 2 since likely all time steps will require a similar amount of wall time to complete. Assuming 2 is chosen, the total time of simulation to be completed will be 1\*Dt+2\*Nrestart\*Dt where Nrestart is the number of solution restarts specified in the script. The total number of time steps (and hence the number of output files generated) will therefore be 1+2\*Nrestart. The value for the time resource request should be chosen so the initial time step and subsequent time steps will complete comfortably within the Slurm time window specifiable up to a maximum of \"#SBATCH \--time=07-00:00\" days.

`<tabs>`{=html} `<tab name="Multinode (by node + restart)">`{=html} `{{File
|name=script-flu-bynode+restart.sh
|lang="bash"
|contents=
#!/bin/bash

<!--T:3402-->
#SBATCH --account=def-group   # Specify account
#SBATCH --time=07-00:00       # Specify time limit dd-hh:mm
#SBATCH --nodes=1             # Specify number of compute nodes (narval 1 node max)
#SBATCH --ntasks-per-node=32  # Specify number of cores per node (max all cores of a compute node)
#SBATCH --mem=0               # Do not change (allocates all memory per compute node)
#SBATCH --cpus-per-task=1     # Do not change
#SBATCH --array=1-5%1         # Specify number of solution restarts (2 or more, 5 is shown)

<!--T:2407-->
module load StdEnv/2023       # Do not change
module load ansys/2023R2      # Specify version (or newer)

<!--T:4403-->
MYVERSION=3d                        # Specify 2d, 2ddp, 3d or 3ddp
MYJOUFILE=sample.jou                # Specify your journal filename
MYJOUFILERES=sample-restart.jou     # Specify journal restart filename
MYCASFILERES=sample-restart.cas.h5  # Specify cas restart filename
MYDATFILERES=sample-restart.dat.h5  # Specify dat restart filename

<!--T:508-->
# ------- do not change any lines below --------

<!--T:4745-->
if [[ "${CC_CLUSTER}" == narval ]]; then
 if [ "$EBVERSIONGENTOO" == 2020 ]; then
   module load intel/2021 intelmpi
   export INTELMPI_ROOT=$I_MPI_ROOT/mpi/latest
   export HCOLL_RCACHE=^ucs
 elif [ "$EBVERSIONGENTOO" == 2023 ]; then
   module load intel/2023 intelmpi
   export INTELMPI_ROOT=$I_MPI_ROOT
 fi
 unset I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS
 unset I_MPI_ROOT
fi

<!--T:4746-->
slurm_hl2hl.py --format ANSYS-FLUENT > /tmp/machinefile-$SLURM_JOB_ID
NCORES=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE * SLURM_CPUS_PER_TASK))

<!--T:3408-->
# Specify 2d, 2ddp, 3d or 3ddp and replace sample with your journal filename …
if [ "$SLURM_NNODES" == 1 ]; then
  #export I_MPI_HYDRA_BOOTSTRAP=ssh
  if [ "$SLURM_ARRAY_TASK_ID" == 1 ]; then
    fluent -g 2ddp -t $NCORES -affinity=0 -i $MYJOUFILE
  else
    fluent -g 2ddp -t $NCORES -affinity=0 -i $MYJOUFILERES
  fi
else 
  if [ "$SLURM_ARRAY_TASK_ID" == 1 ]; then
   if [[ "${CC_CLUSTER}" == nibi ]]; then
     fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -peth -cnf=/tmp/machinefile-$SLURM_JOB_ID -ssh -i $MYJOUFILE
   else
     fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -ssh -i $MYJOUFILE
   fi
  else
   if [[ "${CC_CLUSTER}" == nibi ]]; then
     fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -peth -cnf=/tmp/machinefile-$SLURM_JOB_ID -ssh -i $MYJOUFILERES
   else
     fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -ssh -i $MYJOUFILERES
   fi
  fi
fi
if [ $? -eq 0 ]; then
    echo
    echo "SLURM_ARRAY_TASK_ID  = $SLURM_ARRAY_TASK_ID"
    echo "SLURM_ARRAY_TASK_COUNT = $SLURM_ARRAY_TASK_COUNT"
    echo
    if [ $SLURM_ARRAY_TASK_ID -lt $SLURM_ARRAY_TASK_COUNT ]; then
      echo "Restarting job with the most recent output dat file …"
      ln -sfv output/$(ls -ltr output  grep .cas  tail -n1  awk '{print $9}') $MYCASFILERES
      ln -sfv output/$(ls -ltr output  grep .dat  tail -n1  awk '{print $9}') $MYDATFILERES
      ls -lh cavity* output/*
    else
      echo "Job completed successfully! Exiting now."
      scancel $SLURM_ARRAY_JOB_ID
     fi
else
     echo "Simulation failed. Exiting …"
fi
}}`{=mediawiki} `</tab>`{=html}

`<tab name="Multinode (by core + restart)">`{=html} `{{File
|name=script-flu-bycore+restart.sh
|lang="bash"
|contents=
#!/bin/bash

<!--T:3902-->
#SBATCH --account=def-group   # Specify account
#SBATCH --time=00-03:00       # Specify time limit dd-hh:mm
##SBATCH --nodes=1            # Uncomment to specify (narval 1 node max)
#SBATCH --ntasks=16           # Specify total number of cores
#SBATCH --mem-per-cpu=4G      # Specify memory per core
#SBATCH --cpus-per-task=1     # Do not change
#SBATCH --array=1-5%1         # Specify number of restart aka time steps (2 or more, 5 is shown)

<!--T:3906-->
module load StdEnv/2023       # Do not change
module load ansys/2023R2      # Specify version (or newer)

<!--T:4747-->
MYVERSION=3d                        # Specify 2d, 2ddp, 3d or 3ddp
MYJOUFILE=sample.jou                # Specify your journal filename
MYJOUFILERES=sample-restart.jou     # Specify journal restart filename
MYCASFILERES=sample-restart.cas.h5  # Specify cas restart filename
MYDATFILERES=sample-restart.dat.h5  # Specify dat restart filename

<!--T:509-->
# ------- do not change any lines below --------

<!--T:4748-->
if [[ "${CC_CLUSTER}" == narval ]]; then
 if [ "$EBVERSIONGENTOO" == 2020 ]; then
   module load intel/2021 intelmpi
   export INTELMPI_ROOT=$I_MPI_ROOT/mpi/latest
   export HCOLL_RCACHE=^ucs
 elif [ "$EBVERSIONGENTOO" == 2023 ]; then
   module load intel/2023 intelmpi
   export INTELMPI_ROOT=$I_MPI_ROOT
 fi
 unset I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS
 unset I_MPI_ROOT
fi

<!--T:4749-->
slurm_hl2hl.py --format ANSYS-FLUENT > /tmp/machinefile-$SLURM_JOB_ID
NCORES=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))

<!--T:3910-->
if [ "$SLURM_NNODES" == 1 ]; then
  #export I_MPI_HYDRA_BOOTSTRAP=ssh
  if [ "$SLURM_ARRAY_TASK_ID" == 1 ]; then
    fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pshmem -I $MYFILEJOU
  else
    fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pshmem -I $MYFILEJOURES
  fi
else 
  if [ "$SLURM_ARRAY_TASK_ID" == 1 ]; then
    if [[ "${CC_CLUSTER}" == nibi ]]; then
      fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -peth -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOUFILE
    else
      fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOUFILE
    fi
  else
    if [[ "${CC_CLUSTER}" == nibi ]]; then
      fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -peth -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOUFILERES
    else
      fluent -g $MYVERSION -t $NCORES -affinity=0 -mpi=intel -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOUFILERES
    fi
  fi
fi
if [ $? -eq 0 ]; then
    echo
    echo "SLURM_ARRAY_TASK_ID  = $SLURM_ARRAY_TASK_ID"
    echo "SLURM_ARRAY_TASK_COUNT = $SLURM_ARRAY_TASK_COUNT"
    echo
    if [ $SLURM_ARRAY_TASK_ID -lt $SLURM_ARRAY_TASK_COUNT ]; then
      echo "Restarting job with the most recent output dat file"
      ln -sfv output/$(ls -ltr output  grep .cas  tail -n1  awk '{print $9}') $MYCASFILERES
      ln -sfv output/$(ls -ltr output  grep .dat  tail -n1  awk '{print $9}') $MYDATFILERES
      ls -lh cavity* output/*
    else
      echo "Job completed successfully! Exiting now."
      scancel $SLURM_ARRAY_JOB_ID
     fi
else
     echo "Simulation failed. Exiting now."
fi
}}`{=mediawiki} `</tab>`{=html} `</tabs>`{=html}

### Journal files {#journal_files}

Fluent journal files can include basically any command from Fluent\'s Text-User-Interface (TUI); commands can be used to change simulation parameters like temperature, pressure and flow speed. With this you can run a series of simulations under different conditions with a single case file, by only changing the parameters in the journal file. Refer to the Fluent User\'s Guide for more information and a list of all commands that can be used. The following journal files are set up with `/file/cff-files no` to use the legacy .cas/.dat file format (the default in module versions 2019R3 or older). Set this instead to `/file/cff-files yes` to use the more efficient .cas.h5/.dat.h5 file format (the default in module versions 2020R1 or newer).

`<tabs>`{=html} `<tab name="Journal file (steady, case)">`{=html}

`</tab>`{=html}

`<tab name="Journal file (steady, case + data)">`{=html}

`</tab>`{=html}

`<tab name="Journal file (transient)">`{=html}

`</tab>`{=html}

`</tabs>`{=html}

### UDFs

The first step is to transfer your User-Defined Function or UDF (namely the sampleudf.c source file and any additional dependency files) to the cluster. When uploading from a windows machine, be sure the text mode setting of your transfer client is used otherwise fluent won\'t be able to read the file properly on the cluster since it runs linux. The UDF should be placed in the directory where your journal, cas and dat files reside. Next add one of the following commands into your journal file before the commands that read in your simulation cas/dat files. Regardless of whether you use the Interpreted or Compiled UDF approach, before uploading your cas file onto the Alliance please check that neither the Interpreted UDFs Dialog Box or the UDF Library Manager Dialog Box are configured to use any UDF; this will ensure that only the journal file commands are in control when jobs are submitted.

#### Interpreted

To tell fluent to interpret your UDF at runtime, add the following command line into your journal file before the cas/dat files are read or initialized. The filename sampleudf.c should be replaced with the name of your source file. The command remains the same regardless if the simulation is being run in serial or parallel. To ensure the UDF can be found in the same directory as the journal file, open your cas file in the fluent gui, remove any managed definitions and resave it. Doing this will ensure only the following command/method is in control when fluent runs. To use an interpreted UDF with parallel jobs, it will need to be parallelized as described in the section below.

define/user-defined/interpreted-functions \"sampleudf.c\" \"cpp\" 10000 no

#### Compiled

To use this approach, your UDF must be compiled on an Alliance cluster at least once. Doing so will create a libudf subdirectory structure containing the required `libudf.so` shared library. The libudf directory cannot simply be copied from a remote system (such as your laptop) to the Alliance since the library dependencies of the shared library will not be satisfied, resulting in fluent crashing on startup. That said, once you have compiled your UDF on an Alliance cluster, you can transfer the newly created libudf to any other Alliance cluster, providing your account loads the same StdEnv environment module version. Once copied, the UDF can be used by uncommenting the second (load) libudf line below in your journal file when submitting jobs to the cluster. Both (compile and load) libudf lines should not be left uncommented in your journal file when submitting jobs on the cluster, otherwise your UDF will automatically (re)compiled for each and every job. Not only is this highly inefficient, but it will also lead to racetime-like build conflicts if multiple jobs are run from the same directory. Besides configuring your journal file to build your UDF, the fluent gui (run on any cluster compute node or gra-vdi) may also be used. To do this, you would navigate to the Compiled UDFs Dialog Box, add the UDF source file and click Build. When using a compiled UDF with parallel jobs, your source file should be parallelized as discussed in the section below.

define/user-defined/compiled-functions compile libudf yes sampleudf.c \"\" \"\"

and/or

define/user-defined/compiled-functions load libudf

#### Parallel

Before a UDF can be used with a fluent parallel job (single node SMP and multinode MPI) it will need to be parallelized. By doing this we control how/which processes (host and/or compute) run specific parts of the UDF code when fluent is run in parallel on the cluster. The instrumenting procedure involves adding compiler directives, predicates and reduction macros into your working serial UDF. Failure to do so will result in fluent running slow at best or immediately crashing at worst. The end result will be a single UDF that runs efficiently when fluent is used in both serial and parallel mode. The subject is described in detail under `<I>`{=html}Part I: Chapter 7: Parallel Considerations`</I>`{=html} of the Ansys 2024 `<I>`{=html}Fluent Customization Manual`</I>`{=html} which can be accessed [here](https://docs.alliancecan.ca/Ansys#Online_Documentation "wikilink").

#### DPM

UDFs can be used to customize Discrete Phase Models (DPM) as described in `<I>`{=html}Part III: Solution Mode \| Chapter 24: Modeling Discrete Phase \| 24.2 Steps for Using the Discrete Phase Models\| 24.2.6 User-Defined Functions`</I>`{=html} of the `<I>`{=html}2024R2 Fluent Users Guide`</I>`{=html} and section `<I>`{=html}Part I: Creating and Using User Defined Functions \| Chapter 2: DEFINE Macros \| 2.5 Discrete Phase Model (DPM) DEFINE Macros`</I>`{=html} of the `<I>`{=html}2024R2 Fluent Customization Manual`</I>`{=html}. Before a DMP based UDF can be worked into a simulation, the injection of a set of particles must be defined by specifying \"Point Properties\" with variables such as source position, initial trajectory, mass flow rate, time duration, temperature and so forth depending on the injection type. This can be done in the gui by clicking the Physics panel, Discrete Phase to open the `<I>`{=html}Discrete Phase Model`</I>`{=html} box and then clicking the `<I>`{=html}Injections`</I>`{=html} button. Doing so will open an `<I>`{=html}Injections`</I>`{=html} dialog box where one or more injections can be created by clicking the `<I>`{=html}Create`</I>`{=html} button. The \"Set Injection Properties\" dialog which appears will contain an \"Injection Type\" pulldown with first four types available are \"single, group, surface, flat-fan-atomizer\". If you select any of these then you can then the \"Point Properties\" tab can be selected to input the corresponding Value fields. Another way to specify the \"Point Properties\" would be to read an injection text file. To do this select \"file\" from the Injection Type pulldown, specify the Injection Name to be created and then click the `<I>`{=html}File`</I>`{=html} button (located beside the `<I>`{=html}OK`</I>`{=html} button at the bottom of the \"Set Injection Properties\" dialog). Here either an Injection Sample File (with .dpm extension) or a manually created injection text file can be selected. To Select the File in the Select File dialog box that change the File of type pull down to All Files (\*), then highlight the file which could have any arbitrary name but commonly does have a .inj extension, click the OK button. Assuming there are no problems with the file, no Console error or warning message will appear in fluent. As you will be returned to the \"Injections\" dialog box, you should see the same Injection name that you specified in the \"Set Injection Properties\" dialog and be able to List its Particles and Properties in the console. Next open the Discrete Phase Model Dialog Box and select Interaction with Continuous Phase which will enable updating DPM source terms every flow iteration. This setting can be saved in your cas file or added via the journal file as shown. Once the injection is confirmed working in the gui the steps can be automated by adding commands to the journal file after solution initialization, for example:

`/define/models/dpm/interaction/coupled-calculations yes`\
`/define/models/dpm/injections/delete-injection injection-0:1`\
`/define/models/dpm/injections/create injection-0:1 no yes file no zinjection01.inj no no no no`\
`/define/models/dpm/injections/list-particles injection-0:1`\
`/define/models/dpm/injections/list-injection-properties injection-0:1`

where a basic manually created injection steady file format might look like:

` $ cat  zinjection01.inj`\
` (z=4 12)`\
` ( x          y        z    u         v    w    diameter  t         mass-flow  mass  frequency  time name )`\
` (( 2.90e-02  5.00e-03 0.0 -1.00e-03  0.0  0.0  1.00e-04  2.93e+02  1.00e-06   0.0   0.0        0.0 ) injection-0:1 )`

noting that injection files for DPM simulations are generally setup for either steady or unsteady particle tracking where the format of the former is described in subsection `<I>`{=html}Part III: Solution Mode \| Chapter 24: Modeling Discrete Phase \| 24.3. Setting Initial Conditions for the Discrete Phase \| 24.3.13 Point Properties for File Injections \| 24.3.13.1 Steady File Format`</I>`{=html} of the `<I>`{=html}2024R2 Fluent Customization Manual`</I>`{=html}.

## Ansys CFX {#ansys_cfx}

### Slurm scripts {#slurm_scripts_1}

A summary of command-line options can be printed by running `<b>`{=html}cfx5solve -help`</b>`{=html} where the same module version thats loaded in your slurm script should be first manually loaded. By default cfx5solve will run in single precision (-single). To run cfx5solve in double precision add the `-double` option noting that doing so will also double memory requirements. By default cfx5solve can support meshes with up to 80 million elements (structured) or 200 million elements (unstructured). For larger meshes with up to 2 billion elements, add the `-large` option. Various combinations of these options can be specified for the Partitioner, Interpolator or Solver. Consult the ANSYS CFX-Solver Manager User\'s Guide for further details.

`<tabs>`{=html} `<tab name="Single node">`{=html} `</tab>`{=html}

`<tab name="Multinode">`{=html} `</tab>`{=html}

`</tabs>`{=html}

## Workbench

Before submitting a project file to the queue on a cluster (for the first time) follow these steps to initialize it.\
\# Connect to the cluster with [TigerVNC](https://docs.alliancecan.ca/VNC#Compute_nodes "wikilink").

1.  Switch to the directory where the project file is located (YOURPROJECT.wbpj) and [start Workbench](https://docs.alliancecan.ca/Ansys#Workbench_3 "wikilink") with the same Ansys module you used to create your project.
2.  In Workbench, open the project with `<I>`{=html}File -\> Open`</I>`{=html}.
3.  In the main window, right-click on `<i>`{=html}Setup`</i>`{=html} and select `<I>`{=html}Clear All Generated Data`</I>`{=html}.
4.  In the top menu bar pulldown, select `<I>`{=html}File -\> Exit`</I>`{=html} to exit Workbench.
5.  In the Ansys Workbench popup, when asked `<I>`{=html}The current project has been modified. Do you want to save it?`</I>`{=html}, click on the `<i>`{=html}No`</i>`{=html} button.
6.  Quit Workbench and submit your job using one of the Slurm scripts shown below.

To avoid writing the solution when a running job successfully completes remove `;Save(Overwrite=True)` from within the last line of your script. Doing this will make it easier to run multiple test jobs (for scaling purposes when changing ntasks), since the initialized solution will not be overwritten each time. Alternatively, keep a copy of the initialized YOURPROJECT.wbpj file and YOURPROJECT_files subdirectory and restore them after the solution is written.

### Slurm scripts {#slurm_scripts_2}

A project file can be submitted to the queue by customizing one of the following scripts and then running the `sbatch script-wbpj-202X.sh` command:

`<tabs>`{=html} `<tab name="Single node (StdEnv/2023)">`{=html}

`</tab>`{=html} `<tab name="Single node (StdEnv/2020)">`{=html}

`</tab>`{=html} `</tabs>`{=html}

## Mechanical

The input file can be generated from within your interactive Workbench Mechanical session by clicking `<i>`{=html}Solution -\> Tools -\> Write Input Files`</i>`{=html} then specify `File name:` YOURAPDLFILE.inp and `Save as type:` APDL Input Files (\*.inp). APDL jobs can then be submitted to the queue by running the `sbatch script-name.sh` command.

### Slurm scripts {#slurm_scripts_3}

In the following slurm scripts, lines beginning with `##SBATCH` are commented.

`<tabs>`{=html} `<tab name="Shared Memory Parallel (cpu)">`{=html}

`</tab>`{=html} `<tab name="Distributed Memory Parallel (cpu)">`{=html}

`</tab>`{=html} `<tab name="Shared Memory Parallel (gpu)">`{=html}

`</tab>`{=html} `<tab name="Distributed Memory Parallel (gpu)">`{=html}

`</tab>`{=html} `</tabs>`{=html}

Ansys allocates 1024 MB total memory and 1024 MB database memory by default for APDL jobs. These values can be manually specified (or changed) by adding arguments `-m 1024` and/or `-db 1024` to the mapdl command line in the above scripts. When using a remote institutional license server with multiple Ansys licenses, it may be necessary to add `-p aa_r` or `-ppf anshpc`, depending on which Ansys module you are using. As always, perform detailed scaling tests before running production jobs to ensure that the optimal number of cores and minimum amount memory is specified in your scripts. The `<i>`{=html}single node`</i>`{=html} (SMP Shared Memory Parallel) scripts will typically perform better than the `<i>`{=html}multinode`</i>`{=html} (DIS Distributed Memory Parallel) scripts and therefore should be used whenever possible. To help avoid compatibility issues the Ansys module loaded in your script should ideally match the version used to generate the input file:

`[gra-login2:~/testcase] cat YOURAPDLFILE.inp | grep version`\
`! ANSYS input file written by Workbench version 2019 R3`

## Ansys ROCKY {#ansys_rocky}

Besides being able to run simulations in gui mode (as discussed in the Graphical usage section below) [Ansys Rocky](https://www.ansys.com/products/fluids/ansys-rocky) can also run simulations in non-gui mode. Both modes support running Rocky with cpus only or with cpus and [gpus](https://www.ansys.com/blog/mastering-multi-gpu-ansys-rocky-software-enhancing-its-performance). In the below section two sample slurm scripts are provided where each script would be submitted to the graham queue with the sbatch command as per usual. At the time of this writing neither script has been tested and therefore extensive customization will likely be required. It\'s important to note that these scripts are only usable on graham since the rocky module which they both load is only (at the present time) installed on graham (locally).

### Slurm scripts {#slurm_scripts_4}

To get a full listing of command line options run `Rocky -h` on the command line after loading any rocky module (currently only ansysrocky/2023R2 is available on Graham). If Rocky is being run with gpus to solving coupled problems, the number of cpus you should request from slurm (on the same node) should be increased to a maximum until the scalability limit of the coupled application is reached. If however Rocky is being run with gpus to solve standalone uncoupled problems, then only a minimal number of cpus should be requested that will allow be sufficient for Rocky to still run optimally. For instance only 2cpus or possibly 3cpus may be required. When Rocky is run with \>= 4 cpus then `<I>`{=html}rocky_hpc`</I>`{=html} licenses will be required which the SHARCNET license does provide.

`<tabs>`{=html} `<tab name="CPU only">`{=html}

`</tab>`{=html} `<tab name="GPU based">`{=html}

`</tab>`{=html} `</tabs>`{=html}

# Graphical use {#graphical_use}

Ansys programs may be run interactively in GUI mode on cluster compute nodes or Graham VDI Nodes.

## Compute nodes {#compute_nodes}

Ansys can be run interactively on a single compute node for up to 24 hours. This approach is ideal for testing large simulations since all cores and memory can be requested with salloc as described in [TigerVNC](https://docs.alliancecan.ca/VNC#Compute_Nodes "wikilink"). Once connected with vncviewer, any of the following program versions can be started after loading the required modules as shown below.

### Fluids

:   

    :   `module load StdEnv/2020 ansys/2021R2`
    :   `module load StdEnv/2023 ansys/2022R2` (or newer versions)
    :   `fluent -mpi=intel`, or,
    :   `QTWEBENGINE_DISABLE_SANDBOX=1 cfx5`

### Mapdl

:   

    :   `module load StdEnv/2020 ansys/2021R2`
    :   `module load StdEnv/2023 ansys/2022R2` (or newer versions)
    :   `mapdl -g`, or via launcher,
    :   `launcher` \--\> click RUN button

### Workbench {#workbench_1}

:   

    :   `module load StdEnv/2020 ansys/2021R2`
    :   `module load StdEnv/2023 ansys/2022R2` (or newer versions)
    :   `xfwm4 --replace &` (only needed if using Ansys Mechanical)
    :   `export QTWEBENGINE_DISABLE_SANDBOX=1` (only needed if using CFD-Post)
    :   `runwb2`
    :   \
    :   NOTES :When running an Analysis Program such as Mechanical or Fluent in parallel on a single node, untick `<i>`{=html}Distributed`</i>`{=html} and specify a value of cores equal to your `<b>`{=html}salloc session setting minus 1`</b>`{=html}. The pulldown menus in the Ansys Mechanical workbench do not respond properly. As a workaround run `xfwm4 --replace` on the command line before starting workbench as shown. To make xfwm4 your default edit `$HOME/.vnc/xstartup` and change `mate-session` to `xfce4-session`. Lastly, fluent from ansys/2022R2 does not currently work on compute nodes please use a different version.

### Ensight

:   

    :   `module load StdEnv/2023 ansys/2022R2; A=222; B=5.12.6`, or
    :   `module load StdEnv/2020 ansys/2022R1; A=221; B=5.12.6`, or
    :   `module load StdEnv/2020 ansys/2021R2; A=212; B=5.12.6`, or
    :   `module load StdEnv/2020 ansys/2021R1; A=211; B=5.12.6`
    :   `export LD_LIBRARY_PATH=$EBROOTANSYS/v$A/CEI/apex$A/machines/linux_2.6_64/qt-$B/lib`
    :   `ensight -X`

Note: ansys/2022R2 Ensight is lightly tested on compute nodes. Please let us know if you find any problems using it.

### Rocky

:   

    :   `module load ansysrocky/2023R2 StdEnv/2020 ansys/2023R2`
    :   `module load ansysrocky/2024R2.0 StdEnv/2023 ansys/2024R2.04`
    :   `module load StdEnv/2023 ansys/2025R1`
    :   `Rocky` The ansys module handles reading your \~/licenses/ansys.lic\
    :   `Rocky-gui` Provided by local ansysrocky modules option to select the CMC or SHARCNET server\
    :   `RockySolver` Run rocky solver directly from command line (add -h for help, untested)
    :   `RockySchedular` Start rocky schedular gui to submit/run jobs on present node (untested)
    :   o Rocky for versions 2024R2 or older is only available on gra-vdi and graham clusters (install on all clusters expected in June)
    :   o Rocky for versions 2025R1 and greater is bundled in the ansys module on all clusters (not yet supported by SHARCNET license server)
    :   o The SHARCNET license includes Rocky and is therefore free for all researchers to use
    :   o Rocky supports GPU-accelerated computing however this capability not been tested or documented yet
    :   o To request a graham compute node for interactive use with 4 cpus and 1 gpu for a time limit of 8hrs run:
    :   `salloc --time=08:00:00 --nodes=1 --cpus-per-task=4 --gres=gpu:v100:1 --mem=32G --account=someaccount`

## VDI nodes {#vdi_nodes}

Ansys programs can be run for up to 7days on grahams VDI nodes (gra-vdi.alliancecan.ca) using 8 cores (16 cores max) and 128GB memory. The VDI System provides GPU OpenGL acceleration therefore it is ideal for performing tasks that benefit from high performance graphics. One might use VDI to create or modify simulation input files, post-process data or visualize simulation results. To log in connect with [TigerVNC](https://docs.alliancecan.ca/VNC#VDI_Nodes "wikilink") then open a new terminal window and start one of the program versions shown below. The vertical bar `|` notation is used to separate the various commands. The maximum job size for any parallel job run on gra-vdi should be limited to 16cores to avoid overloading the servers and impacting other users. To run two simultaneous gui jobs (16 cores max each) on gra-vdi establish two independent vnc sessions. For the first session, connect to gra-vdi3.sharcnet.ca with vnc. For the second session connect to gra-vdi4.sharecnet.ca also with vnc. Then within each session, start Ansys in gui mode and run your simulation. Note that simultaneous simulations should in general be run in different directories to avoid file conflict issues. Unlike compute nodes vnc connections (which impose slurm limits through salloc) there is no time limit constraint on gra-vdi when running simulations.

### Fluids {#fluids_1}

:   

    :   `module load CcEnv StdEnv/2023`
    :   `ansys/2024R2.04` (or older versions)
    :   `unset SESSION_MANAGER`
    :   `fluent | cfx5 | icemcfd`
    :   o Where unsetting SESSION_MANAGER prevents the following Qt message from appearing when starting fluent:
    :   \[`<span style="Color:#ff7f50">`{=html}Qt: Session management error: None of the authentication protocols specified are supported`</span>`{=html}\]
    :   o In the event the following message appears in a popup window when starting icemcfd \...
    :   \[`<span style="Color:#ff7f50">`{=html}Error segmentation violation - exiting after doing an emergency save`</span>`{=html}\]
    :   \... do not click the popup OK button otherwise icemcfd will crash. Instead do the following (one time only):
    :   click the Settings Tab -\> Display -\> tick X11 -\> Apply -\> OK -\> File -\> Exit
    :   The error popup should no longer appear when icemcfd is restarted.

### Mapdl {#mapdl_1}

:   

    :   `module load CcEnv StdEnv/2023`
    :   `ansys/2024R2.04` (or older versions)
    :   `mapdl -g` (to start the gui directly), or,
    :   `unset SESSION_MANAGER; launcher -> click RUN button`

### Workbench {#workbench_2}

:   

    :   `module load SnEnv`
    :   `module load ansys/2024R2.04` (or older versions)
    :   `runwb2` or `runwb2-gui` (see NOTE1 below)
    :   

        ------------------------------------------------------------------------

```{=html}
<!-- -->
```

:   

    :   `module load CcEnv StdEnv/2023`
    :   `module load ansys/2025R1` (or newer versions)
    :   `runwb2` (see NOTE5 below)
    :   

        ------------------------------------------------------------------------

```{=html}
<!-- -->
```

:   

    :   NOTE1: The `runwb2` command (for locally installed) ansys modules provided by `SnEnv` will use the SHARCNET license server by default, as set in the ansys module file that you load. To use a remote server, start workbench with `runwb2-gui` instead as this wrapper script will read your `~/.licenses/ansys.lic` file similar to modules available under `StdEnv/2023` on the clusters. In addition an interactive option to use the CMC server (routed through the SHARCNET CMC CadPASS server) will be offered, thus removing the need to configure it in your ansys.lic file.
    :   NOTE2: When starting fluent from within workbench with version 2025R1 before clicking the start button, click the Environment tab in the Fluent Launcher panel and copy/paste HOOPS_PICTURE=opengl in the empty input box. Optionally set `export HOOPS_PICTURE=opengl` in your environment before starting workbench. Doing either of these will eliminate the following red warning that will other wise appear in the tui startup messages: \[`<span style="Color:#ff7f50">`{=html}Warning: Software rasterizer found, hardware acceleration will be disabled.`</span>`{=html}\]
    :   NOTE3: When running Mechanical in Workbench on gra-vdi be sure to `<b>`{=html}tic`</b>`{=html} `<i>`{=html}Distributed`</I>`{=html} in the upper ribbon Solver panel and specify a maximum value of `<b>`{=html}24`</b>`{=html} cores. When running Fluent on gra-vdi instead, `<b>`{=html}untic`</b>`{=html} `<i>`{=html}Distributed`</I>`{=html} and specify a maximum value of `<b>`{=html}12`</b>`{=html} cores. Do not attempt to use more than 128GB memory otherwise Ansys will hit the hard limit and be killed. If you need more cores or memory then please use a cluster compute node to run your graphical session on (as described in the previous Compute nodes section above). When doing old pre-processing or post-processing work with Ansys on gra-vdi and not running calculation, please only use `<b>`{=html}4`</b>`{=html} cores otherwise hpc licenses will be checked out unnecessarily.
    :   NOTE4: On rare occasions Ansys workbench or some programs that run within it will freeze or not start properly; Inparticular if vncviewer unexpectedly disconnected before Ansys could be cleanly shutdown. In general, if Ansys is not behaving properly open a new terminal window on gra-vdi and run `pkill -9 -e -u $USER -f "ansys|fluent|mwrpcss|mwfwrapper|ENGINE|mono"` to fully kill off all Ansys processes. If the problem persists then try running `rm -rf .ansys` if you have been working in the gui on compute nodes before working on gra-vdi. If the problem involves home, project or scratch (the df command hangs) then its highly probable ansys will resume working normally again once the attached storage issue is resolved.
    :   NOTE5: The ansys/2025R1 module is not usable with the SHARCNET license server until an upgrade is complete expected early august/2025. On gra-vdi module versions ansys/2025 or newer are not installed under SnEnv and may only be used under CcEnv. The gui interphase for some programs available in ansys/2025R1 such as Ansys Mechanical do not work properly on gra-vdi and will not be fixed as gra-vdi is to be retired soon.

### Ensight {#ensight_1}

:   

    :   `module load SnEnv`
    :   `ansys/2024R2.04` (or older versions back to 2021R2)
    :   `ensight`\

### Rocky {#rocky_1}

:   

    :   `module load clumod ansysrocky/2023R2 CcEnv StdEnv/2020 ansys/2023R2`, or,
    :   `module load clumod ansysrocky/2024R2.0 CcEnv StdEnv/2023 ansys/2024R2.04`, or,
    :   `module load CcEnv StdEnv/2023 ansys/2025R1`
    :   `Rocky` The ansys module handles reading your \~/licenses/ansys.lic\
    :   `Rocky-gui` Provided by local ansysrocky modules option to select the CMC or SHARCNET server\
    :   `RockySolver` Run rocky solver directly from command line (add -h for help, untested)
    :   `RockySchedular` Start rocky schedular gui to submit/run jobs on present node (untested)
    :   o Rocky for versions 2024R2 or older is only available on gra-vdi and graham clusters (install on all clusters expected in June)
    :   o Rocky for versions 2025R1 and greater is bundled in the ansys module on all clusters (not yet supported by SHARCNET license server)
    :   o Rocky should only be used with cpus and not gpus when run on gra-vdi since the machine only has one gpu intended for graphics
    :   o The SHARCNET license includes Rocky and is therefore free for all researchers across the Alliance to use
    :   o The Rocky Innovation Space is located [<https://innovationspace.ansys.com/ais-rocky/>](https://innovationspace.ansys.com/ais-rocky/)
    :   o For Ansys Rocky 2024 R2 and 2025 R1 Release Highlights click [here](https://innovationspace.ansys.com/knowledge/forums/topic/ansys-rocky-2024-r2-release-highlights/) and [here](https://innovationspace.ansys.com/knowledge/forums/topic/ansys-rocky-2025-r1-release-highlights/)

## SSH issues {#ssh_issues}

:   

    :   Some Ansys GUI programs can be run remotely on a cluster compute node by X forwarding over SSH to your local desktop. Unlike VNC, this approach is untested and unsupported since it relies on a properly setup X display server for your particular operating system OR the selection, installation and configuration of a suitable X client emulator package such as MobaXterm. Most users will find interactive response times unacceptably slow for basic menu tasks let alone performing more complex tasks such as those involving graphics rendering. Startup times for GUI programs can also be very slow depending on your Internet connection. For example, in one test it took 40 minutes to fully start the gui up over SSH while starting it with vncviewer required only 34 seconds. Despite the potential slowness when connecting over SSH to run GUI programs, doing so may still be of interest if your only goal is to open a simulation and perform some basic menu operations or run some calculations. The basic steps are given here as a starting point: 1) ssh -Y username@graham.computecanada.ca; 2) salloc \--x11 \--time=1:00:00 \--mem=16G \--cpus-per-task=4 \[\--gpus-per-node=1\] \--account=def-mygroup; 3) once connected onto a compute node try running `xclock`. If the clock appears on your desktop, proceed to load the desired Ansys module and try running the program.

# Site-specific usage {#site_specific_usage}

## SHARCNET license {#sharcnet_license}

The SHARCNET Ansys license is free for academic use by `<b>`{=html}any`</b>`{=html} Alliance researcher on `<b>`{=html}any`</b>`{=html} Alliance system. The installed software does not have any solver or geometry limits. The SHARCNET license may be used for `<b>`{=html}`<i>`{=html}Publishable Academic Research`</i>`{=html}`</b>`{=html} but not for any private/commercial purposes as this is strictly prohibited by the license terms. The SHARCNET Ansys license is based on the Multiphysics Campus Solution and includes products such as: HF, EM, Electronics HPC, Mechanical, CFD as described [here](https://www.ansys.com/academic/educator-tools/academic-product-portfolio). ROCKY and LS-DYNA are also now included by the SHARCNET license. Lumerical acquired by ANSYS in 2020 however is not covered presently although the software is installed with recent ansys modules so can be used with other suitably licensed Ansys servers. SpaceClaim is not installed Alliance systems (since they are all linux based) however it is technically covered by the SHARCNET license. A pool of 1986 anshpc licenses is included with the SHARCNET license to support running large scale parallel simulations with most Ansys products. To ensure they are used efficiently scaling tests should be run before launching long jobs. Parallel jobs that do not achieve at least 50% CPU utilization will probably be flagged by the system, resulting in a followup by an Alliance team member.

The SHARCNET Ansys license is made available on a first come first serve basis. It currently permits each researcher to run a maximum of simultaneous 8 jobs using upto 512 hpc cores. Therefore any of the following maximum even sized combinations can be run simultaneously 1x512, 2x256, 4x128 or 8x64 across all clusters. Since the license is oversubscribed there is however the potential for a shortage of anshpc licenses to develop. Should a job fail on startup due to a shortage of licenses it will need to be manually be resubmitted. If over time there are many instances of license shortages reported then either the total job limit per researcher will be decreased (to 6 or 4) and/or the total hpc core limit per researcher will be decreased (to 384 cores or 256) if necessary. If you need more than 512 hpc cores for your research then consider using the local ANSYS License server at your institution if one is available and contributing towards expanding it if necessary.

Some researchers may prefer to purchase a license subscription from [CMC](https://www.cmc.ca/subscriptions/) to gain access to their remote license servers to run ansys anywhere besides just on Alliance systems such as in your lab or at home on your laptop. Doing so will have several benefits 1) a local institutional license server is not needed 2) a physical license does not need to be obtained and reconfigured each year 3) the license can be used [almost anywhere](https://www.cmc.ca/ansys-campus-solutions-cmc-00200-04847/) including at home, institutions, or any alliance cluster across Canada and 4) installation instructions are provided for Windows machines to enable running spaceclaim (not currently possible on the Alliance clusters since all systems are linux based). Note however that according to the CMC [Ansys Quick Start Guides](https://www.cmc.ca/qsg-ansys-cadpass-r20/) there may be a 64 core limit per user!

#### License file {#license_file}

To use the SHARCNET Ansys license on any Alliance cluster, simply configure your `ansys.lic` file as follows:

``` bash
[username@cluster:~] cat ~/.licenses/ansys.lic
setenv("ANSYSLMD_LICENSE_FILE", "1055@license3.sharcnet.ca")
setenv("ANSYSLI_SERVERS", "2325@license3.sharcnet.ca")
```

#### License query {#license_query}

To show the number of licenses in use by your username and the total in use by all users, run:

``` bash
ssh graham.computecanada.ca
module load ansys
lmutil lmstat -c $ANSYSLMD_LICENSE_FILE -a | grep "Users of\|$USER"
```

If you discover any licenses unexpectedly in use by your username (usually due to ansys not exiting cleanly on gra-vdi), connect to the node where it\'s running, open a terminal window and run the following command to terminate the rogue processes `pkill -9 -e -u $USER -f "ansys"` after which your licenses should be freed. Note that gra-vdi consists of two nodes (gra-vdi3 and gra-vdi4) which researchers are randomly placed on when connecting to gra-vdi.computecanada.ca with [TigerVNC](https://docs.alliancecan.ca/VNC#VDI_Nodes "wikilink"). Therefore it\'s necessary to specify the full hostname (gra-vdi3.sharcnet.ca or grav-vdi4.sharcnet.ca) when connecting with tigervnc to ensure you log into the correct node before running pkill.

### Local modules {#local_modules}

When using gra-vdi, researchers have the choice of loading Ansys modules from our global environment (after loading CcEnv) or loading Ansys modules installed locally on the machine itself (after loading SnEnv). The local modules may be of interest as they include some Ansys programs and versions not yet supported by the standard environment. When starting programs from local Ansys modules, you can select the CMC license server or continue to use the SHARCNET license server by default. Settings from `~/.licenses/ansys.lic` are only used when `dash gui` is appended to the ansys program name for instance `<b>`{=html}`fluent-gui``</b>`{=html} instead of simply `<b>`{=html}`fluent``</b>`{=html}. Suitable usage of Ansys on gra-vdi : run a single job interactively (in the gui or from the command line) with up to 8 cores and 128G RAM, create or modify simulation input files, post process or visualize data. To load and use a local ansys module on gra-vdi do the following:

1.  Connect to gra-vdi.computecanada.ca with [TigerVNC](https://docs.alliancecan.ca/VNC#VDI_Nodes "wikilink").
2.  Open a new terminal window and load a module:
3.  `<b>`{=html}`module load SnEnv ansys/2024R2``</b>`{=html} (or older)
4.  Directly start one of the following ansys programs from the command line:
5.  `<b>`{=html}`runwb2``</b>`{=html}`[-gui]|``<b>`{=html}`fluent``</b>`{=html}`[-gui]|``<b>`{=html}`cfx5``</b>`{=html}`[-gui]``</b>`{=html}`|``<b>`{=html}`icemcfd``</b>`{=html}`[-gui]|``<b>`{=html}`apdl``</b>`{=html}`[-gui]|``<b>`{=html}`Rocky``</b>`{=html}`[-gui]`

If you run `<b>`{=html}`cfx5-gui``</b>`{=html} your `~/.licenses/ansys.lic` file will first be read and then you will get the option to instead select the CMC server and finally choose which CFX program will be started in gui mode from the following :

`   1) CFX-Launcher  (cfx5 -> cfx5launch)`\
`   2) CFX-Pre       (cfx5pre)`\
`   3) CFD-Post      (cfdpost -> cfx5post)`\
`   4) CFX-Solver    (cfx5solve)`

License feature preferences previously setup with `<i>`{=html}anslic_admin`</i>`{=html} are no longer supported (2021-09-09). If a license problem occurs, try removing the `~/.ansys` directory in your /home account to clear the settings. If problems persist please contact our [technical support](https://docs.alliancecan.ca/technical_support "wikilink") and provide the contents of your `~/.licenses/ansys.lic` file.

# Additive Manufacturing {#additive_manufacturing}

To get started configure your `~/.licenses/ansys.lic` file to point to a license server that has a valid Ansys Mechanical License. This must be done on all systems where you plan to run the software.

## Enable Additive {#enable_additive}

This section describes how to make the Ansys Additive Manufacturing ACT extension available for use in your project. The steps must be performed on each cluster for each ansys module version where the extension will be used. Any extensions needed by your project will also need to be installed on the cluster as described below. If you get warnings about missing un-needed extensions (such as ANSYSMotion) then uninstall them from your project.

### Download Extension {#download_extension}

-   download AdditiveWizard.wbex from <https://catalog.ansys.com/>
-   upload AdditiveWizard.wbex to the cluster where it will be used

### Start Workbench {#start_workbench}

-   follow the Workbench section in [Graphical use above](https://docs.alliancecan.ca/ANSYS#Graphical_use "wikilink").
-   File -\> Open your project file (ending in .wbpj) into Workbench gui

### Open Extension Manager {#open_extension_manager}

-   click ACT Start Page and the ACT Home page tab will open
-   click Manage Extensions and the Extension Manager will open

### Install Extension {#install_extension}

-   click the box with the large + sign under the search bar
-   navigate to select and install your AdditiveWizard.wbex file

### Load Extension {#load_extension}

-   click to highlight the AdditiveWizard box (loads the AdditiveWizard extension for current session only)
-   click lower right corner arrow in the AdditiveWizard box and select `<i>`{=html}Load extension`</i>`{=html} (loads the extension for current AND future sessions)

### Unload Extension {#unload_extension}

-   click to un-highlight the AdditiveWizard box (unloads extension for the current session only)
-   click lower right corner arrow in the AdditiveWizard box and select `<I>`{=html}Do not load as default`</i>`{=html} (extension will not load for future sessions)

## Run Additive {#run_additive}

### Gra-vdi {#gra_vdi}

A user can run a single Ansys Additive Manufacturing job on gra-vdi with up to 16 cores as follows:

-   Start Workbench on Gra-vdi as described above in `<b>`{=html}Enable Additive`</b>`{=html}.
-   click File -\> Open and select `<i>`{=html}test.wbpj`</i>`{=html} then click Open
-   click View -\> reset workspace if you get a grey screen
-   start Mechanical, Clear Generated Data, tick Distributed, specify Cores
-   click File -\> Save Project -\> Solve

Check utilization:

-   open another terminal and run: `top -u $USER` \*\*OR\*\* `ps u -u $USER | grep ansys`
-   kill rogue processes from previous runs: `pkill -9 -e -u $USER -f "ansys|mwrpcss|mwfwrapper|ENGINE"`

Please note that rogue processes can persistently tie up licenses between gra-vdi login sessions or cause other unusual errors when trying to start gui programs on gra-vdi. Although rare, rogue processes can occur if an ansys gui session (fluent, workbench, etc) is not cleanly terminated by the user before vncviewer is terminated either manually or unexpectedly - for instance due to a transient network outage or hung filesystem. If the latter is to blame then the processes may not by killable until normal disk access is restored.

### Cluster

Project preparation:

Before submitting a newly uploaded Additive project to a cluster queue (with `sbatch scriptname`) certain preparations must be done. To begin, open your simulation with Workbench gui (as described in the `Enable Additive` section above) in the same directory that your job will be submitted from and then save it again. Be sure to use the same ansys module version that will be used for the job. Next create a Slurm script (as explained in the `<i>`{=html}Cluster Batch Job Submission - WORKBENCH`</I>`{=html} section above). To perform parametric studies, change `Update()` to `UpdateAllDesignPoints()` in the Slurm script. Determine the optimal number of cores and memory by submitting several short test jobs. To avoid needing to manually clear the solution `<b>`{=html}and`</b>`{=html} recreate all the design points in Workbench between each test run, either 1) change `Save(Overwrite=True)` to `Save(Overwrite=False)` or 2) save a copy of the original YOURPROJECT.wbpj file and corresponding YOURPROJECT_files directory. Optionally create and then manually run a replay file on the cluster in the respective test case directory between each run, noting that a single replay file can be used in different directories by opening it in a text editor and changing the internal FilePath setting.

`module load ansys/2019R3`\
`rm -f test_files/.lock`\
`runwb2 -R myreplay.wbjn`

Resource utilization:

Once your additive job has been running for a few minutes, a snapshot of its resource utilization on the compute node(s) can be obtained with the following srun command. Sample output corresponding to an eight core submission script is shown next. It can be seen that two nodes were selected by the scheduler:

`[gra-login1:~] srun --jobid=myjobid top -bn1 -u $USER | grep R | grep -v top`\
`  PID USER   PR  NI    VIRT    RES    SHR S  %CPU %MEM    TIME+  COMMAND`\
`22843 demo   20   0 2272124 256048  72796 R  88.0  0.2  1:06.24  ansys.e`\
`22849 demo   20   0 2272118 256024  72822 R  99.0  0.2  1:06.37  ansys.e`\
`22838 demo   20   0 2272362 255086  76644 R  96.0  0.2  1:06.37  ansys.e`\
`  PID USER   PR  NI    VIRT    RES    SHR S  %CPU %MEM    TIME+  COMMAND`\
` 4310 demo   20   0 2740212 271096 101892 R 101.0  0.2  1:06.26  ansys.e`\
` 4311 demo   20   0 2740416 284552  98084 R  98.0  0.2  1:06.55  ansys.e`\
` 4304 demo   20   0 2729516 268824 100388 R 100.0  0.2  1:06.12  ansys.e`\
` 4305 demo   20   0 2729436 263204 100932 R 100.0  0.2  1:06.88  ansys.e`\
` 4306 demo   20   0 2734720 431532  95180 R 100.0  0.3  1:06.57  ansys.e`

Scaling tests:

After a job completes, its \"Job Wall-clock time\" can be obtained from `seff myjobid`. Using this value, scaling tests can be performed by submitting short test jobs with an increasing number of cores. If the Wall-clock time decreases by \~50% when the number of cores is doubled, additional cores may be considered.

# Help resources {#help_resources}

## Online documentation {#online_documentation}

A publicly accessible online version of the [Ansys Help site](https://ansyshelp.ansys.com/public/account/secured?returnurl=/Views/Secured/prod_page.html?pn=Fluent&pid=Fluent&lang=en) with full Documentation, Tutorial and Videos for the `<b>`{=html}LATEST`</b>`{=html} Ansys release version. Developer documentation is available [here](https://developer.ansys.com/docs) also without login. To access documentation for Ansys versions 2024R1, 2024R2 AND 2025R1 you however WILL need to login [here](https://ansyshelp.ansys.com/). There is documentation for Multiphysics System Coupling versions 2023R1 \-\--\> 202R2 available [here](https://ansysapi.ansys.com) where again login will be required. To access the Ansys Help for a specific version such as 2023R2 perform the below steps precisely which should work for any module installed under StdEnv/2023 (thus back to 2022R2) and \*\*NOT\*\* require login:

1.  Connect to `<b>`{=html}gra-vdi.computecanada.ca`</b>`{=html} with tigervnc as described [here](https://docs.alliancecan.ca/VNC#VDI_Nodes "wikilink").
2.  If the Firefox browser or the Ansys Workbench is open, close it now.
3.  Start Firefox by clicking `<I>`{=html}Applications -\> Internet -\> Firefox`</I>`{=html}.
4.  Open a `<b>`{=html}`<i>`{=html}new`</I>`{=html}`</b>`{=html} terminal window by clicking `<I>`{=html}Applications -\> System Tools -\> Mate Terminal`</I>`{=html}.
5.  Start Workbench by typing the following in your terminal: `<i>`{=html}module load CcEnv StdEnv/2023 ansys; runwb2`</i>`{=html}
6.  Go to the upper Workbench menu bar and click `<I>`{=html}Help -\> ANSYS Workbench Help`</I>`{=html}. The `<b>`{=html}Workbench Users\' Guide`</b>`{=html} should appear loaded in Firefox.
7.  At this point Workbench is no longer needed so close it by clicking the `<I>`{=html}\>Unsaved Project - Workbench`</I>`{=html} tab located along the bottom frame (doing this will bring Workbench into focus) and then click `<I>`{=html}File -\> Exit`</I>`{=html}.
8.  In the top middle of the Ansys documentation page, click the word `<I>`{=html}HOME`</I>`{=html} located just left of `<I>`{=html}API DOCS`</I>`{=html}.
9.  Now scroll down and you should see a list of Ansys product icons and/or alphabetical ranges.
10. Select a product to view its documentation. The documentation for the latest release version will be displayed by default. Change the version by clicking the `<I>`{=html}Release Year`</I>`{=html} pull down located above and just to the right of the Ansys documentation page search bar.
11. To search for documentation corresponding to a different Ansys product, click `<I>`{=html}HOME`</I>`{=html} again.

## YouTube videos {#youtube_videos}

Search the Ansys How To videos [here](https://www.youtube.com/@AnsysHowTo/videos).

## Innovation Space {#innovation_space}

Search the Ansys Innovation Space [here](https://innovationspace.ansys.com/).

`</translate>`{=html}
