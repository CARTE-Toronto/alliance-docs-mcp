---
title: "Abaqus/en"
url: "https://docs.alliancecan.ca/wiki/Abaqus/en"
category: "General"
last_modified: "2025-10-21T17:29:34Z"
page_id: 10030
display_title: "Abaqus"
---

`<languages />`{=html}

\_\_FORCETOC\_\_ [Abaqus FEA](https://www.3ds.com/products-services/simulia/products/abaqus/) is a software suite for finite element analysis and computer-aided engineering.

# Using your own license {#using_your_own_license}

Abaqus software modules are available on our clusters; however, you must provide your own license. To configure your account on a cluster, log in and create a file named `$HOME/.licenses/abaqus.lic` containing the following line. Next, replace `port@server` with the flexlm port number and server IP address (or fully qualified hostname) of your Abaqus license server. If you want to use legacy version 6.14.1 then replace ABAQUSLM_LICENSE_FILE with LM_LICENSE_FILE.

If your license has not been set up for use on an Alliance cluster, some additional configuration changes by the Alliance system administrator and your local system administrator will need to be done. Such changes are necessary to ensure the flexlm and vendor TCP ports of your Abaqus server are reachable from all cluster compute nodes when jobs are run via the queue. For us to help you get this done, write to [technical support](https://docs.alliancecan.ca/Technical_support "technical support"){.wikilink}. Please be sure to include the following three items:

- flexlm port number
- static vendor port number
- IP address of your Abaqus license server

You will then be sent a list of cluster IP addresses so that your administrator can open the local server firewall to allow connections from the cluster on both ports. Please note that a special license agreement must generally be negotiated and signed by SIMULIA and your institution before a local license may be used remotely on Alliance hardware.

# Cluster job submission {#cluster_job_submission}

Below are prototype Slurm scripts for submitting thread and mpi-based parallel simulations to single or multiple compute nodes. Most users will find it sufficient to use one of the `<b>`{=html}project directory scripts`</b>`{=html} provided in the `<i>`{=html}Single node computing`</i>`{=html} sections. The optional `memory=` argument found in the last line of the scripts is intended for larger memory or problematic jobs where 3072MB offset value may require tuning. A listing of all Abaqus command line arguments can be obtained by loading an Abaqus module and running: `abaqus -help | less`.

Single node jobs that run less than one day should find the `<i>`{=html}project directory script`</i>`{=html} located in the first tab sufficient. However, single node jobs that run for more than a day should use one of the restart scripts. Jobs that create large restart files will benefit by writing to the local disk through the use of the SLURM_TMPDIR environment variable utilized in the `<b>`{=html}temporary directory scripts`</b>`{=html} provided in the two rightmost tabs of the single node standard and explicit analysis sections. The restart scripts shown here will continue jobs that have been terminated early for some reason. Such job failures can occur if a job reaches its maximum requested runtime before completing and is killed by the queue or if the compute node the job was running on crashed due to an unexpected hardware failure. Other restart types are possible by further tailoring of the input file (not shown here) to continue a job with additional steps or change the analysis (see the documentation for version specific details).

Jobs that require large memory or larger compute resources (beyond that which a single compute node can provide) should use the mpi scripts in the `<b>`{=html}multiple node sections`</b>`{=html} below to distribute computing over arbitrary node ranges determined automatically by the scheduler. Short scaling test jobs should be run to determine wall-clock times (and memory requirements) as a function of the number of cores (2, 4, 8, etc.) to determine the optimal number before running any long jobs.

## Standard analysis {#standard_analysis}

Abaqus solvers support thread-based and mpi-based parallelization. Scripts for each type are provided below for running Standard Analysis type jobs on Single or Multiple nodes respectively. Scripts to perform multiple node job restarts are not currently provided.

### Single node computing {#single_node_computing}

`<tabs>`{=html} `<tab name="project directory script">`{=html} `{{File
  |name="scriptsp1.txt"
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --account=def-group    # Specify account
#SBATCH --time=00-06:00        # Specify days-hrs:mins
#SBATCH --cpus-per-task=4      # Specify number of cores
#SBATCH --mem=8G               # Specify total memory > 5G
#SBATCH --nodes=1              # Do not change !
##SBATCH --constraint=cascade  # Uncomment to specify node (cpu/gpu jobs)
##SBATCH --gres=gpu:t4:1       # Uncomment to specify gpu
# or
##SBATCH --constraint=rome     # Uncomment to specify node (gpu only jobs)
##SBATCH --gres=gpu:a100:1     # Uncomment to specify gpu

module load StdEnv/2020        # Latest installed version
module load abaqus/2021        # Latest installed version

#module load StdEnv/2016       # Uncomment to use
#module load abaqus/2020       # Uncomment to use

unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE=$LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE=$ABAQUSLM_LICENSE_FILE"

rm -f testsp1* testsp2*
abaqus job=testsp1 input=mystd-sim.inp \
   scratch=$SLURM_TMPDIR cpus=$SLURM_CPUS_ON_NODE interactive \
   mp_mode=threads memory="$((${SLURM_MEM_PER_NODE}-3072))MB" \
#  gpus=$SLURM_GPUS_ON_NODE  # uncomment this line to use gpu
}}`{=mediawiki}

To write restart data every N=12 time increments specify in the input file:

`*RESTART, WRITE, OVERLAY, FREQUENCY=12`

To write restart data for a total of 12 time increments specify instead:

`*RESTART, WRITE, OVERLAY, NUMBER INTERVAL=12, TIME MARKS=NO`

To check for completed restart information do:

`egrep -i "step|start" testsp*.com testsp*.msg testsp*.sta`

Some simulations may benefit by adding the following to the Abaqus command at the bottom of the script:

`order_parallel=OFF`

`</tab>`{=html} `<tab name="project directory restart script">`{=html} `{{File
  |name="scriptsp2.txt"
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --account=def-group    # Specify account
#SBATCH --time=00-06:00        # Specify days-hrs:mins
#SBATCH --cpus-per-task=4      # Specify number of cores
#SBATCH --mem=8G               # Specify total memory > 5G
#SBATCH --nodes=1              # Do not change !
##SBATCH --constraint=cascade  # Uncomment to specify node (44cores)
##SBATCH --gres=gpu:t4:1       # Uncomment to specify gpu
# or
##SBATCH --constraint=rome     # Uncomment to specify node (128cores)
##SBATCH --gres=gpu:a100:1     # Uncomment to specify gpu

module load StdEnv/2020        # Latest installed version
module load abaqus/2021        # Latest installed version

unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE=$LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE=$ABAQUSLM_LICENSE_FILE"

rm -f testsp2* testsp1.lck
abaqus job=testsp2 oldjob=testsp1 input=mystd-sim-restart.inp \
   scratch=$SLURM_TMPDIR cpus=$SLURM_CPUS_ON_NODE interactive \
   mp_mode=threads memory="$((${SLURM_MEM_PER_NODE}-3072))MB" \
#  gpus=$SLURM_GPUS_ON_NODE  # uncomment this line to use gpu
}}`{=mediawiki}

The restart input file should contain:

`*HEADING`\
`*RESTART, READ`

`</tab>`{=html} `<tab name="temporary directory script">`{=html} `{{File
  |name="scriptst1.txt"
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --account=def-group    # Specify account
#SBATCH --time=00-06:00        # Specify days-hrs:mins
#SBATCH --cpus-per-task=4      # Specify number of cores
#SBATCH --mem=8G               # Specify total memory > 5G
#SBATCH --nodes=1              # Do not change !
##SBATCH --constraint=cascade  # Uncomment to specify node (cpu/gpu jobs)
##SBATCH --gres=gpu:t4:1       # Uncomment to specify gpu
# or
##SBATCH --constraint=rome     # Uncomment to specify node (gpu only jobs)
##SBATCH --gres=gpu:a100:1     # Uncomment to specify gpu

module load StdEnv/2020        # Latest installed version
module load abaqus/2021        # Latest installed version

unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE=$LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE=$ABAQUSLM_LICENSE_FILE"
echo "SLURM_SUBMIT_DIR =" $SLURM_SUBMIT_DIR
echo "SLURM_TMPDIR = " $SLURM_TMPDIR

rm -f testst1* testst2*
mkdir $SLURM_TMPDIR/scratch
cd $SLURM_TMPDIR
while sleep 6h; do
   echo "Saving data due to time limit ..."
   cp -fv * $SLURM_SUBMIT_DIR 2>/dev/null
done &
WPID=$!
abaqus job=testst1 input=$SLURM_SUBMIT_DIR/mystd-sim.inp \
   scratch=$SLURM_TMPDIR/scratch cpus=$SLURM_CPUS_ON_NODE interactive \
   mp_mode=threads memory="$((${SLURM_MEM_PER_NODE}-3072))MB" \
#  gpus=$SLURM_GPUS_ON_NODE  # uncomment this line to use gpu
{ kill $WPID && wait $WPID; } 2>/dev/null
cp -fv * $SLURM_SUBMIT_DIR
}}`{=mediawiki}

To write restart data every N=12 time increments specify in the input file:

`*RESTART, WRITE, OVERLAY, FREQUENCY=12`

To write restart data for a total of 12 time increments specify instead:

`*RESTART, WRITE, OVERLAY, NUMBER INTERVAL=12, TIME MARKS=NO`

To check the completed restart information do:

`egrep -i "step|start" testst*.com testst*.msg testst*.sta`

`</tab>`{=html} `<tab name="temporary directory restart script">`{=html} `{{File
  |name="scriptst2.txt"
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --account=def-group    # Specify account
#SBATCH --time=00-06:00        # Specify days-hrs:mins
#SBATCH --cpus-per-task=4      # Specify number of cores
#SBATCH --mem=8G               # Specify total memory > 5G
#SBATCH --nodes=1              # Do not change !
##SBATCH --constraint=cascade  # Uncomment to specify node (44 cores)
##SBATCH --gres=gpu:t4:1       # Uncomment to specify gpu
# or
##SBATCH --constraint=rome     # Uncomment to specify node (128 cores)
##SBATCH --gres=gpu:a100:1     # Uncomment to specify gpu

module load StdEnv/2020        # Latest installed version
module load abaqus/2021        # Latest installed version

unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE=$LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE=$ABAQUSLM_LICENSE_FILE"
echo "SLURM_SUBMIT_DIR =" $SLURM_SUBMIT_DIR
echo "SLURM_TMPDIR = " $SLURM_TMPDIR

rm -f testst2* testst1.lck
cp testst1* $SLURM_TMPDIR
mkdir $SLURM_TMPDIR/scratch
cd $SLURM_TMPDIR
while sleep 6h; do
   echo "Saving data due to time limit ..."
   cp -fv testst2* $SLURM_SUBMIT_DIR 2>/dev/null
done &
WPID=$!
abaqus job=testst2 oldjob=testst1 input=$SLURM_SUBMIT_DIR/mystd-sim-restart.inp \
   scratch=$SLURM_TMPDIR/scratch cpus=$SLURM_CPUS_ON_NODE interactive \
   mp_mode=threads memory="$((${SLURM_MEM_PER_NODE}-3072))MB" \
#  gpus=$SLURM_GPUS_ON_NODE  # uncomment this line to use gpu
{ kill $WPID && wait $WPID; } 2>/dev/null
cp -fv testst2* $SLURM_SUBMIT_DIR
}}`{=mediawiki}

The restart input file should contain:

`*HEADING`\
`*RESTART, READ`

`</tab>`{=html} `</tabs>`{=html}

### Multiple node computing {#multiple_node_computing}

Users with large memory or compute needs (and correspondingly large licenses) can use the following script to perform mpi-based computing over an arbitrary range of nodes ideally left to the scheduler to automatically determine. A companion template script to perform restart of multinode jobs is not provided due to additional limitations when they can be used.

`xargs)"`

for i in \`echo \"\$nodes\" xargs -n1 uniq\`; do hostlist=\${hostlist}\$(echo \"\[\'\${i}\',\$(echo \"\$nodes\" xargs -n1 grep \$i wc -l)\],\"); done hostlist=\"\$(echo \"\$hostlist\" sed \'s/,\$//g\')\" mphostlist=\"mp_host_list=\[\$(echo \"\$hostlist\")\]\" export \$mphostlist echo \"\$mphostlist\" \> abaqus_v6.env

abaqus job=testsp1-mpi input=mystd-sim.inp \\

` scratch=$SLURM_TMPDIR cpus=$SLURM_NTASKS interactive mp_mode=mpi \`\
` #mp_host_split=1  # number of dmp processes per node >= 1 (uncomment to specify)`

}}

## Explicit analysis {#explicit_analysis}

Abaqus solvers support thread-based and mpi-based parallelization. Scripts for each type are provided below for running explicit analysis type jobs on single or multiple nodes respectively. Template scripts to perform multinode job restarts are not currently provided pending further testing.

### Single node computing {#single_node_computing_1}

`<tabs>`{=html} `<tab name="project directory script">`{=html} `{{File
  |name="scriptep1.txt"
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --account=def-group    # specify account
#SBATCH --time=00-06:00        # days-hrs:mins
#SBATCH --mem=8000M            # node memory > 5G
#SBATCH --cpus-per-task=4      # number cores > 1
#SBATCH --nodes=1              # do not change

module load StdEnv/2020
module load abaqus/2021

unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE=$LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE=$ABAQUSLM_LICENSE_FILE"

rm -f testep1* testep2*
abaqus job=testep1 input=myexp-sim.inp \
   scratch=$SLURM_TMPDIR cpus=$SLURM_CPUS_ON_NODE interactive \
   mp_mode=threads memory="$((${SLURM_MEM_PER_NODE}-3072))MB"
}}`{=mediawiki}

To write restart data for a total of 12 time increments specify in the input file:

`*RESTART, WRITE, OVERLAY, NUMBER INTERVAL=12, TIME MARKS=NO`

Check for completed restart information in relevant output files:

`egrep -i "step|restart" testep*.com testep*.msg testep*.sta`

`</tab>`{=html} `<tab name="project directory restart script">`{=html} `{{File
  |name="scriptep2.txt"
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --account=def-group    # specify account
#SBATCH --time=00-06:00        # days-hrs:mins
#SBATCH --mem=8000M            # node memory > 5G
#SBATCH --cpus-per-task=4      # number cores > 1
#SBATCH --nodes=1              # do not change

module load StdEnv/2020
module load abaqus/2021

unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE=$LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE=$ABAQUSLM_LICENSE_FILE"

rm -f testep2* testep1.lck
for f in testep1*; do [[ -f ${f} ]] && cp -a "$f" "testep2${f#testep1}"; done
abaqus job=testep2 input=myexp-sim.inp recover \
   scratch=$SLURM_TMPDIR cpus=$SLURM_CPUS_ON_NODE interactive \
   mp_mode=threads memory="$((${SLURM_MEM_PER_NODE}-3072))MB"
}}`{=mediawiki}

No input file modifications are required to restart the analysis.

`</tab>`{=html} `<tab name="temporary directory script">`{=html} `{{File
  |name="scriptet1.txt"
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --account=def-group    # specify account
#SBATCH --time=00-06:00        # days-hrs:mins
#SBATCH --mem=8000M            # node memory > 5G
#SBATCH --cpus-per-task=4      # number cores > 1
#SBATCH --nodes=1              # do not change

module load StdEnv/2020
module load abaqus/2021

unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE=$LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE=$ABAQUSLM_LICENSE_FILE"
echo "SLURM_SUBMIT_DIR =" $SLURM_SUBMIT_DIR
echo "SLURM_TMPDIR = " $SLURM_TMPDIR

rm -f testet1* testet2*
cd $SLURM_TMPDIR
while sleep 6h; do
   cp -f * $SLURM_SUBMIT_DIR 2>/dev/null
done &
WPID=$!
abaqus job=testet1 input=$SLURM_SUBMIT_DIR/myexp-sim.inp \
   scratch=$SLURM_TMPDIR cpus=$SLURM_CPUS_ON_NODE interactive \
   mp_mode=threads memory="$((${SLURM_MEM_PER_NODE}-3072))MB"
{ kill $WPID && wait $WPID; } 2>/dev/null
cp -f * $SLURM_SUBMIT_DIR
}}`{=mediawiki}

To write restart data for a total of 12 time increments specify in the input file:

`*RESTART, WRITE, OVERLAY, NUMBER INTERVAL=12, TIME MARKS=NO`

Check for completed restart information in relevant output files:

`egrep -i "step|restart" testet*.com testet*.msg testet*.sta`

`</tab>`{=html} `<tab name="temporary directory restart script">`{=html} `{{File
  |name="scriptet2.txt"
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --account=def-group    # specify account
#SBATCH --time=00-06:00        # days-hrs:mins
#SBATCH --mem=8000M            # node memory > 5G
#SBATCH --cpus-per-task=4      # number cores > 1
#SBATCH --nodes=1              # do not change

module load StdEnv/2020
module load abaqus/2021

unset SLURM_GTIDS
export MPI_IC_ORDER='tcp'
echo "LM_LICENSE_FILE=$LM_LICENSE_FILE"
echo "ABAQUSLM_LICENSE_FILE=$ABAQUSLM_LICENSE_FILE"
echo "SLURM_SUBMIT_DIR =" $SLURM_SUBMIT_DIR
echo "SLURM_TMPDIR = " $SLURM_TMPDIR

rm -f testet2* testet1.lck
for f in testet1*; do cp -a "$f" $SLURM_TMPDIR/"testet2${f#testet1}"; done
cd $SLURM_TMPDIR
while sleep 3h; do
   cp -f * $SLURM_SUBMIT_DIR 2>/dev/null
done &
WPID=$!
abaqus job=testet2 input=$SLURM_SUBMIT_DIR/myexp-sim.inp recover \
   scratch=$SLURM_TMPDIR cpus=$SLURM_CPUS_ON_NODE interactive \
   mp_mode=threads memory="$((${SLURM_MEM_PER_NODE}-3072))MB"
{ kill $WPID && wait $WPID; } 2>/dev/null
cp -f  * $SLURM_SUBMIT_DIR
}}`{=mediawiki}

No input file modifications are required to restart the analysis.

`</tab>`{=html} `</tabs>`{=html}

### Multiple node computing {#multiple_node_computing_1}

`xargs)"`

for i in \`echo \"\$nodes\" xargs -n1 uniq\`; do hostlist=\${hostlist}\$(echo \"\[\'\${i}\',\$(echo \"\$nodes\" xargs -n1 grep \$i wc -l)\],\"); done hostlist=\"\$(echo \"\$hostlist\" sed \'s/,\$//g\')\" mphostlist=\"mp_host_list=\[\$(echo \"\$hostlist\")\]\" export \$mphostlist echo \"\$mphostlist\" \> abaqus_v6.env

abaqus job=testep1-mpi input=myexp-sim.inp \\

` scratch=$SLURM_TMPDIR cpus=$SLURM_NTASKS interactive mp_mode=mpi \`\
` #mp_host_split=1  # number of dmp processes per node >= 1 (uncomment to specify)`

}}

## Memory estimates {#memory_estimates}

### Single process {#single_process}

An estimate for the total slurm node memory (\--mem=) required for a simulation to run fully in ram (without being virtualized to scratch disk) can be obtained by examining the Abaqus output `test.dat` file. For example, a simulation that requires a fairly large amount of memory might show:

``` bash
                   M E M O R Y   E S T I M A T E
  
 PROCESS      FLOATING PT       MINIMUM MEMORY        MEMORY TO
              OPERATIONS           REQUIRED          MINIMIZE I/O
             PER ITERATION           (MB)               (MB)
  
     1          1.89E+14             3612              96345
```

Alternatively the total memory estimate for a single node threaded process could be obtained by running the simulation interactively on a compute node and then monitor the memory consumption using the ps or top commands. The follows described how to do the latter:\
1) ssh into a cluster, obtain an allocation on a compute node (such as gra100), and start your simulation running:

2\) ssh into the cluster again, then ssh into the compute node reserved by salloc and run top i.e.

3\) watch the VIRT and RES columns until steady peak memory values are observed

To completely satisfy the recommended \"MEMORY TO OPERATIONS REQUIRED MINIMIZE I/O\" (MRMIO) value, at least the same amount of non-swapped physical memory (RES) must be available to Abaqus. Since the RES will in general be less than the virtual memory (VIRT) by some relatively constant amount for a given simulation, it is necessary to slightly over-allocate the requested Slurm node memory `-mem=`. In the above sample Slurm script, this over-allocation has been hardcoded to a conservative value of 3072MB based on initial testing of the standard Abaqus solver. To avoid long queue wait times associated with large values of MRMIO, it may be worth investigating the simulation performance impact associated with reducing the RES memory that is made available to Abaqus significantly below the MRMIO. This can be done by lowering the `-mem=` value which in turn will set an artificially low value of `memory=` in the Abaqus command (found in the last line of the script). In doing this, the RES cannot dip below the MINIMUM MEMORY REQUIRED (MMR) otherwise Abaqus will exit due to Out of Memory (OOM). As an example, if your MRMIO is 96GB try running a series of short test jobs with `#SBATCH --mem=8G, 16G, 32G, 64G` until an acceptable minimal performance impact is found, noting that smaller values will result in increasingly larger scratch space used by temporary files.

### Multi process {#multi_process}

To determine the required slurm memory for multi-node slurm scripts, memory estimates (per compute process) required to minimize I/O are given in the output dat file of completed jobs. If mp_host_split is not specified (or is set to 1) then the total number of compute processes will equal the number of nodes. The mem-per-cpu value can then be roughly determined by multiplying the largest memory estimate by the number of nodes and then dividing by the number or ntasks. If however a value for mp_host_split is specified (greater than 1) than the mem-per-cpu value can be roughly determined from the largest memory estimate times the number of nodes times the value of mp_host_split divided by the number of tasks. Note that mp_host_split must be less than or equal to the number of cores per node assigned by slurm at runtime otherwise Abaqus will terminate. This scenario can be controlled by uncommenting to specify a value for tasks-per-node. The following definitive statement is given in every output dat file and mentioned here for reference:

` THE UPPER LIMIT OF MEMORY THAT CAN BE ALLOCATED BY ABAQUS WILL IN GENERAL DEPEND ON THE VALUE OF`\
`THE "MEMORY" PARAMETER AND THE AMOUNT OF PHYSICAL MEMORY AVAILABLE ON THE MACHINE. PLEASE SEE`\
`THE "ABAQUS ANALYSIS USER'S MANUAL" FOR MORE DETAILS. THE ACTUAL USAGE OF MEMORY AND OF DISK`\
`SPACE FOR SCRATCH DATA WILL DEPEND ON THIS UPPER LIMIT AS WELL AS THE MEMORY REQUIRED TO MINIMIZE`\
`I/O. IF THE MEMORY UPPER LIMIT IS GREATER THAN THE MEMORY REQUIRED TO MINIMIZE I/O, THEN THE ACTUAL`\
`MEMORY USAGE WILL BE CLOSE TO THE ESTIMATED "MEMORY TO MINIMIZE I/O" VALUE, AND THE SCRATCH DISK`\
`USAGE WILL BE CLOSE-TO-ZERO; OTHERWISE, THE ACTUAL MEMORY USED WILL BE CLOSE TO THE PREVIOUSLY`\
`MENTIONED MEMORY LIMIT, AND THE SCRATCH DISK USAGE WILL BE ROUGHLY PROPORTIONAL TO THE DIFFERENCE`\
`BETWEEN THE ESTIMATED "MEMORY TO MINIMIZE I/O" AND THE MEMORY UPPER LIMIT. HOWEVER ACCURATE`\
`ESTIMATE OF THE SCRATCH DISK SPACE IS NOT POSSIBLE.`

# Graphical use {#graphical_use}

It is now recommended to use OpenOnDemand or JupyterLab to run graphical applications at the Alliance.

## OnDemand

1\. Connect to an OnDemand system using one of the following URLs in your laptop browser :\
[NIBI](https://docs.alliancecan.ca/wiki/Nibi#Access_through_Open_OnDemand_(OOD)): [`https://ondemand.sharcnet.ca`](https://ondemand.sharcnet.ca)

`FIR: `[`https://jupyterhub.fir.alliancecan.ca`](https://jupyterhub.fir.alliancecan.ca)\
`RORQUAL: `[`https://jupyterhub.rorqual.alliancecan.ca`](https://jupyterhub.rorqual.alliancecan.ca)\
`TRILLIUM: `[`https://ondemand.scinet.utoronto.ca`](https://ondemand.scinet.utoronto.ca)

2\. Open a new terminal window within your desktop and load one of :

`module load StdEnv/2020 abaqus/2021, or`\
`module load StdEnv/2023 abaqus/2025 <- coming soon`

3\. Start the application in graphical mode:\
abaqus cae

To start Abaqus in gui mode there must be at least `<b>`{=html}one`</b>`{=html} unused cae license according to :

`$ abaqus licensing lmstat -c $ABAQUSLM_LICENSE_FILE -a | grep "Users of cae"`\
`Users of cae:  (Total of 4 licenses issued;  Total of 3 licenses in use)`

## VncViewer

1\. Connect with a VncViewer client to a login or compute node by following [TigerVNC](https://docs.alliancecan.ca/VNC "TigerVNC"){.wikilink}\
2. Open a new terminal window and enter the following\
module load StdEnv/2020 abaqus/2021, or

`module load StdEnv/2023 abaqus/2025 <- coming soon`

3\. Start the application with\
abaqus cae -mesa

# Site-specific use {#site_specific_use}

## SHARCNET license {#sharcnet_license}

The SHARCNET license has been renewed until 17-jan-2026. It provides a small but free license consisting of 2 cae and 35 execute tokens where usage limits are imposed 10 tokens/user and 15 tokens/group. For groups that have purchased dedicated tokens, the free token usage limits are added to their reservation. The free tokens are available on a first come first serve basis and mainly intended for testing and light usage before deciding whether or not to purchase dedicated tokens. Costs for dedicated tokens (in 2021) were approximately CAD\$110 per compute token and CAD\$400 per GUI token: submit a ticket to request an official quote. The license can be used by any Alliance researcher, but only on SHARCNET hardware. Groups that purchase dedicated tokens to run on the SHARCNET license server may likewise only use them on SHARCNET hardware including the SHARCNET [OOD](https://docs.alliancecan.ca/wiki/Nibi#Access_through_Open_OnDemand_(OOD)) system (to run Abaqus in graphical mode) or Nibi/Dusky clusters (for submitting compute batch jobs to the queue). Before you can use the license, you must contact [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink} and request access. In your email 1) mention that it is for use on SHARCNET systems and 2) include a copy/paste of the following `License Agreement` statement with your full name and username entered in the indicated locations. Please note that every user must do this it cannot be done one time only for a group; this includes PIs who have purchased their own dedicated tokens.

### License agreement {#license_agreement}

    ----------------------------------------------------------------------------------
    Subject: Abaqus SHARCNET Academic License User Agreement

    This email is to confirm that i "_____________" with username "___________" will
    only use “SIMULIA Academic Software” with tokens from the SHARCNET license server
    for the following purposes:

    1) on SHARCNET hardware where the software is already installed
    2) in affiliation with a Canadian degree-granting academic institution
    3) for education, institutional or instruction purposes and not for any commercial
       or contract-related purposes where results are not publishable
    4) for experimental, theoretical and/or digital research work, undertaken primarily
       to acquire new knowledge of the underlying foundations of phenomena and observable
       facts, up to the point of proof-of-concept in a laboratory    
    -----------------------------------------------------------------------------------

### Configure license file {#configure_license_file}

Configure your license file as follows, noting that it is only usable on SHARCNET systems such as nibi and dusky clusters or the SHARCNET OOD desktop system.

``` bash
[l2 (nibi login node):~] cat ~/.licenses/abaqus.lic
prepend_path("ABAQUSLM_LICENSE_FILE","27050@license3.sharcnet.ca")
```

If your Abaqus job fails with the error message \[\*\*\* ABAQUS/eliT_CheckLicense rank 0 terminated by signal 11 (Segmentation fault)\] then verify your `abaqus.lic` file contains ABAQUSLM_LICENSE_FILE when using abaqus/202X modules. If your Abaqus jobs fails with error message \[License server machine is down or not responding, etc.\] and you are using abaqus/6.14.1 then replace ABAQUSLM_LICENSE_FILE with LM_LICENSE_FILE.

### Query license server {#query_license_server}

Log into nibi cluster, load abaqus and then run one of the following:

``` bash
ssh nibi.alliancecan.ca
module load StdEnv/2020
module load abaqus
```

I\) Check the SHARCNET license server for started and queued jobs:

``` bash
abaqus licensing lmstat -c $ABAQUSLM_LICENSE_FILE -a | egrep "Users|start|queued"
```

II\) Check the SHARCNET license server for started and queued jobs also showing reservations by purchasing groups:

``` bash
abaqus licensing lmstat -c $ABAQUSLM_LICENSE_FILE -a | egrep "Users|start|queued|RESERVATION"
```

III\) Check the SHARCNET license server for only cae, standard and explicit product availability:

``` bash
abaqus licensing lmstat -c $ABAQUSLM_LICENSE_FILE -a | grep "Users of" | egrep "cae|standard|explicit"
```

When the output of query I) above indicates that a job for a particular username is queued this means the job has entered the \"R\"unning state from the perspective of `squeue -j jobid` or `sacct -j jobid` and is therefore idle on a compute node waiting for a license. This will have the same impact on your account priority as if the job were performing computations and consuming CPU time. Eventually when sufficient licenses come available the queued job will start.

#### Example

The following shows the situation where a user submitted two 6-core jobs (each requiring 12 tokens) in quick succession. The scheduler then started each job on a different node in the order they were submitted. Since the user had 10 Abaqus compute tokens, the first job (27527287) was able to acquire exactly enough (10) tokens for the solver to begin running. The second job (27527297) not having access to any more tokens entered an idle \"queued\" state (as can be seen from the lmstat output) until the first job completed, wasting the available resources and depreciating the user\'s fair share level in the process \...

`[l2 (nibi login node):~] sq`\
`           JOBID     USER              ACCOUNT           NAME  ST  TIME_LEFT NODES CPUS TRES_PER_N MIN_MEM NODELIST (REASON) `\
`        27530366  roberpj         cc-debug_cpu  scriptsp2.txt   R    9:56:13     1    6        N/A      8G     c107  (None) `\
`        27530407  roberpj         cc-debug_cpu  scriptsp2.txt   R    9:59:37     1    6        N/A      8G     c292  (None) `

`[l2 (nibi login node):~] abaqus licensing lmstat -c $ABAQUSLM_LICENSE_FILE -a | egrep "Users|start|queued"`\
`Users of abaqus:  (Total of 78 licenses issued;  Total of 53 licenses in use)`\
`   roberpj c107 /dev/tty (v62.6) (license3.sharcnet.ca/27050 1042), start Mon 11/25 17:15, 10 licenses`\
`   roberpj c292 /dev/tty (v62.6) (license3.sharcnet.ca/27050 125) queued for 10 licenses`

To avoid license shortage problems when submitting multiple jobs when working with expensive Abaqus tokens either use a [job dependency](https://docs.alliancecan.ca/wiki/Running_jobs#Cancellation_of_jobs_with_dependency_conditions_which_cannot_be_met), [job array](https://docs.alliancecan.ca/wiki/Job_arrays) or at the very least set up a slurm [email notification](https://docs.alliancecan.ca/wiki/Running_jobs#Email_notification) to know when your job completes before manually submitting another one.

### Specify job resources {#specify_job_resources}

To ensure optimal usage of both your Abaqus tokens and our resources, it\'s important to carefully specify the required memory and ncpus in your Slurm script. The values can be determined by submitting a few short test jobs to the queue then checking their utilization. For `<b>`{=html}completed`</b>`{=html} jobs use `seff JobNumber` to show the total `<i>`{=html}Memory Utilized`</i>`{=html} and `<i>`{=html}Memory Efficiency`</i>`{=html}. If the `<i>`{=html}Memory Efficiency`</i>`{=html} is less than \~90%, decrease the value of the `#SBATCH --mem=` setting in your Slurm script accordingly. Notice that the `seff JobNumber` command also shows the total `<i>`{=html}CPU (time) Utilized`</i>`{=html} and `<i>`{=html}CPU Efficiency`</i>`{=html}. If the `<i>`{=html}CPU Efficiency`</i>`{=html} is less than \~90%, perform scaling tests to determine the optimal number of CPUs for optimal performance and then update the value of `#SBATCH --cpus-per-task=` in your Slurm script. For `<b>`{=html}running`</b>`{=html} jobs, use the `srun --jobid=29821580 --pty top -d 5 -u $USER` command to watch the %CPU, %MEM and RES for each Abaqus parent process on the compute node. The %CPU and %MEM columns display the percent usage relative to the total available on the node while the RES column shows the per process resident memory size (in human readable format for values over 1GB). Further information regarding how to [monitor jobs](https://docs.alliancecan.ca/Running_jobs#Monitoring_jobs "monitor jobs"){.wikilink} is available on our documentation wiki

### Core token mapping {#core_token_mapping}

    TOKENS 5  6  7  8  10  12  14  16  19  21  25  28  34  38
    CORES  1  2  3  4   6   8  12  16  24  32  48  64  96 128

where TOKENS = floor\[5 X CORES\^0.422\]

Each GPU used requires 1 additional TOKEN

## Western license {#western_license}

The Western site license may only be used by Western researchers on hardware located at Western\'s campus. Currently, only the Dusky cluster satisfies this condition. Nibi and SHARCNET OOD system are excluded since they are located on Waterloo\'s campus. Contact the Western Abaqus license server administrator \<jmilner@robarts.ca\> to inquire about using the Western Abaqus license. You will need to provide your username and possibly make arrangements to purchase tokens. If you are granted access then you may proceed to configure your `abaqus.lic` file to point to the Western license server:

### Configure license file {#configure_license_file_1}

``` bash
[dus241:~] cat .licenses/abaqus.lic
prepend_path("LM_LICENSE_FILE","27000@license4.sharcnet.ca")
prepend_path("ABAQUSLM_LICENSE_FILE","27000@license4.sharcnet.ca")
```

Once configured, submit your job as described in the `<i>`{=html}Cluster job submission`</i>`{=html} section above. If there are any problems submit a problem ticket to [technical support](https://docs.alliancecan.ca/Technical_support "technical support"){.wikilink}. Specify that you are using the Abaqus Western license on dusky and provide the failed job number along with a paste of any error messages as applicable.
