---
title: "Niagara Quickstart/en"
url: "https://docs.alliancecan.ca/wiki/Niagara_Quickstart/en"
category: "General"
last_modified: "2025-04-19T05:17:02Z"
page_id: 7034
display_title: "Niagara Quickstart"
---

=Specifications=

The Niagara cluster is a large cluster of 1548 Lenovo SD350 servers each with 40 Intel "Skylake" cores at 2.4 GHz.
The peak performance of the cluster is 3.02 PFlops delivered / 4.75 PFlops theoretical.  It is the 53rd fastest supercomputer on the TOP500 list of June 2018.

Each node of the cluster has 188 GiB / 202 GB RAM per node (at least 4 GiB/core for user jobs).  Being designed for large parallel workloads, it has a fast interconnect consisting of EDR InfiniBand in a Dragonfly+ topology with Adaptive Routing.  The compute nodes are accessed through a queueing system that allows jobs with a minimum of 15 minutes and a maximum of 24 hours and favours large jobs.

See the Intro to Niagara recording

More detailed hardware characteristics of the Niagara supercomputer can be found on this page.

= Getting started on Niagara =

Access to Niagara is not enabled automatically for everyone with an Alliance account, but anyone with an active Alliance account can get their access enabled.

If you have an active Alliance account but you do not have access to Niagara yet (e.g. because you are a new user and belong to a group whose primary PI does not have an allocation as granted in the annual Alliance RAC), go to the opt-in page on the CCDB site.  After clicking the "Join" button on that page, it usually takes only one or two business days for access to be granted.

Please read this document carefully.  The FAQ is also a useful resource.  If at any time you require assistance, or if something is unclear, please do not hesitate to contact us.

== Logging in ==
There are two ways to access Niagara:

# Via your browser with Open OnDemand. This is recommended for users who are not familiar with Linux or the command line. Please see our quickstart guide for more instructions on how to use Open OnDemand.
# Terminal access with ssh. Please read the following instructions.

Niagara runs CentOS 7, which is a type of Linux.  You will need to be familiar with Linux systems to work on Niagara.  If you are not it will be worth your time to review
the Linux introduction or to attend a local "Linux Shell" workshop.

As with all SciNet and Alliance compute systems, access to Niagara is done via ssh (secure shell) only. As of January 22 2022, authentication is only allowed via SSH keys. Please refer to this page to generate your SSH key pair and make sure you use them securely.

Open a terminal window (e.g. PuTTY on Windows or MobaXTerm), then ssh into the Niagara login nodes with your CC credentials:

$ ssh -i /path/to/ssh_private_key -Y MYCCUSERNAME@niagara.scinet.utoronto.ca

or

$ ssh -i /path/to/ssh_private_key -Y MYCCUSERNAME@niagara.computecanada.ca

The Niagara login nodes are where you develop, edit, compile, prepare and submit jobs.

These login nodes are not part of the Niagara compute cluster, but have the same architecture, operating system, and software stack.

The optional -Y is needed to open windows from the Niagara command-line onto your local X server.

To run on Niagara's compute nodes, you must submit a batch job.

If you cannot log in, be sure first to check the System Status on this site's front page.

== Your various directories ==
By virtue of your access to Niagara you are granted storage space on the system.  There are several directories available to you, each indicated by an associated environment variable

=== home and scratch ===

You have a home and scratch directory on the system, whose locations are of the form

$HOME=/home/g/groupname/myccusername

$SCRATCH=/scratch/g/groupname/myccusername

where groupname is the name of your PI's group, and myccusername is your CC username.  For example:
nia-login07:~$ pwd
/home/s/scinet/rzon
nia-login07:~$ cd $SCRATCH
nia-login07:rzon$ pwd
/scratch/s/scinet/rzon

NOTE: home is read-only on compute nodes.

=== project and archive ===

Users from groups with RAC storage allocation will also have a project and possibly an archive directory.

$PROJECT=/project/g/groupname/myccusername

$ARCHIVE=/archive/g/groupname/myccusername

NOTE: Currently archive space is available only via HPSS, and is not accessible on the Niagara login, compute, or datamover nodes.

IMPORTANT: Future-proof your scripts

When writing your scripts, use the environment variables ($HOME, $SCRATCH, $PROJECT, $ARCHIVE) instead of the actual paths!  The paths may change in the future.

=== Storage and quotas ===

You should familiarize yourself with the  various file systems, what purpose they serve, and how to properly use them.  This table summarizes the various file systems.  See the  Data management at Niagara page for more details.

location	quota              	quota                  	block size	expiration time	backed up	on login nodes	on compute nodes
$HOME   	100 GB per user    	100 GB per user        	1 MB      	               	yes      	yes           	read-only
$SCRATCH	25 TB per user     	25 TB per user         	16 MB     	2 months       	no       	yes           	yes
$SCRATCH	50-500TB per group 	depending on group size	16 MB     	2 months       	no       	yes           	yes
$PROJECT	by group allocation	by group allocation    	16 MB     	               	yes      	yes           	yes
$ARCHIVE	by group allocation	by group allocation    	          	               	dual-copy	no            	no
$BBUFFER	10 TB per user     	10 TB per user         	1 MB      	very short     	no       	yes           	yes

=== Moving data to Niagara ===

If you need to move data to Niagara for analysis, or when you need to move data off of Niagara, use the following guidelines:
* If your data is less than 10GB, move the data using the login nodes.
* If your data is greater than 10GB, move the data using the datamover nodes nia-datamover1.scinet.utoronto.ca and nia-datamover2.scinet.utoronto.ca .

Details of how to use the datamover nodes can be found on the  Data management at Niagara page.

= Loading software modules =

You have two options for running code on Niagara: use existing software, or  compile your own.  This section focuses on the former.

Other than essentials, all installed software is made available  using module commands. These modules set environment variables (PATH, etc.), allowing multiple, conflicting versions of a given package to be available.  A detailed explanation of the module system can be  found on the modules page.

Common module subcommands are:
module load : use particular software
module purge: remove currently loaded modules
module spider (or module spider ): list available software packages
module avail: list loadable software packages
module list: list loaded modules

Along with modifying common environment variables, such as PATH, and LD_LIBRARY_PATH, these modules also create a SCINET_MODULENAME_ROOT environment variable, which can be used to access commonly needed software directories, such as /include and /lib.

There are handy abbreviations for the module commands. ml is the same as module list, and ml  is the same as module load .

== Software stacks: NiaEnv and CCEnv ==

On Niagara, there are two available software stacks:

A  Niagara software stack tuned and compiled for this machine. This stack is available by default, but if not, can be reloaded with
module load NiaEnv
The standard Alliance software stack which is available on Alliance's other clusters (including Graham, Cedar, Narval, and Beluga):
module load CCEnv arch/avx512
(without the arch/avx512 module, you'd get the modules for a previous generation of CPUs)
Or, if you want the same default modules loaded as on Cedar, Graham, and Beluga, then do

module load CCEnv arch/avx512 StdEnv/2018.3

== Tips for loading software ==

We advise against loading modules in your .bashrc. This could lead to very confusing behaviour under certain circumstances.

Our guidelines for .bashrc files can be found here

Instead, load modules by hand when needed, or by sourcing a separate script.

Load run-specific modules inside your job submission script.

Short names give default versions; e.g. intel → intel/2018.2.  It is usually better to be explicit about the versions, for future reproducibility.

Modules sometimes require other modules to be loaded first.

Solve these dependencies by using module spider.

= Available compilers and interpreters =

* For most compiled software, one should use the Intel compilers (icc for C, icpc for C++, and ifort for Fortran). Loading an intel module makes these available.
* The GNU compiler suite (gcc, g++, gfortran) is also available, if you load one of the gcc modules.
* Open source interpreted, interactive software is also available:
** Python
** R
** Julia
** Octave

Please visit the Python or R page for details on using these tools.  For information on running MATLAB applications on Niagara, visit  this page.

= Using Commercial Software =

May I use commercial software on Niagara?

* Possibly, but you have to bring your own license for it.  You can connect to an external license server using ssh tunneling.
* SciNet and Alliance have an extremely large and broad user base of thousands of users, so we cannot provide licenses for everyone's favorite software.
* Thus, the only commercial software installed on Niagara is software that can benefit everyone: compilers, math libraries and debuggers.
* That means no MATLAB, Gaussian, IDL,
* Open source alternatives like Octave, Python, and R are available.
* We are happy to help you to install commercial software for which you have a license.
* In some cases, if you have a license, you can use software in the Alliance stack.
The list of commercial software which is installed on Niagara, for which you will need a license to use, can be found on the commercial software page.

= Compiling on Niagara: Example =

Suppose one want to compile an application from two c source files, main.c and module.c, which use the Gnu Scientific Library (GSL). This is an example of how this would be done:
nia-login07:~$ module list
Currently Loaded Modules:
  1) NiaEnv/2018a (S)
  Where:
   S:  Module is Sticky, requires --force to unload or purge

nia-login07:~$ module load intel/2018.2 gsl/2.4

nia-login07:~$ ls
appl.c module.c

nia-login07:~$ icc -c -O3 -xHost -o appl.o appl.c
nia-login07:~$ icc -c -O3 -xHost -o module.o module.c
nia-login07:~$ icc  -o appl module.o appl.o -lgsl -mkl

nia-login07:~$ ./appl

Note:
* The optimization flags -O3 -xHost allow the Intel compiler to use instructions specific to the architecture CPU that is present (instead of for more generic x86_64 CPUs).
* Linking with this library is easy when using the intel compiler, it just requires the -mkl flags.
* If compiling with gcc, the optimization flags would be -O3 -march=native.  For the way to link with the MKL, it is suggested to use the MKL link line advisor.

= Testing =

You really should test your code before you submit it to the cluster to know if your code is correct and what kind of resources you need.

Small test jobs can be run on the login nodes.
Rule of thumb: tests should run no more than a couple of minutes, taking at most about 1-2GB of memory, and use no more than a couple of cores.

You can run the ddt debugger on the login nodes after module load ddt.

Short tests that do not fit on a login node, or for which you need a dedicated node, request an interactive debug job with the debug command:
nia-login07:~$ debugjob N
where N is the number of nodes, If N=1, this gives an interactive session one 1 hour, when N=4 (the maximum), it gives you 30 minutes.  Finally, if your debugjob process takes more than 1 hour, you can request an interactive job from the regular queue using the salloc command.  Note, however, that this may take some time to run, since it will be part of the regular queue, and will be run when the scheduler decides.
nia-login07:~$ salloc --nodes N --time=M:00:00
here N is again the number of nodes, and M is the number of hours you wish the job to run.
If you need to use graphics while testing your code through salloc, e.g. when using a debugger such as DDT or DDD, you have the following options, please visit the
 Testing with graphics page.

= Submitting jobs =

Niagara uses SLURM as its job scheduler.
You submit jobs from a login node by passing a script to the sbatch command:
nia-login07:scratch$ sbatch jobscript.sh
This puts the job in the queue. It will run on the compute nodes in due course.
In most cases, you will want to submit from your $SCRATCH directory, so that the output of your compute job can be written out (as mentioned above, $HOME is read-only on the compute nodes).
Jobs will run under their group's RRG allocation, or, if the group has none, under a RAS allocation (previously called 'default' allocation).
Some example job scripts can be found below.

Keep in mind:

Scheduling is by node, so in multiples of 40-cores.
 Your job's maximum walltime is 24 hours.
Jobs must write to your scratch or project directory (home is read-only on compute nodes).
Compute nodes have no internet access.
 Move your data to Niagara before you submit your job.

== Scheduling by node ==
On many systems that use SLURM, the scheduler will deduce from the specifications of the number of tasks and the number of cpus-per-node what resources should be allocated.  On Niagara things are a bit different.

All job resource requests on Niagara are scheduled as a multiple of nodes.
The nodes that your jobs run on are exclusively yours, for as long as the job is running on them.

No other users are running anything on them.
You can ssh into them to see how things are going.

Whatever your requests to the scheduler, it will always be translated into a multiple of nodes allocated to your job.
Memory requests to the scheduler are of no use. Your job always gets N x 202GB of RAM, where N is the number of nodes and 202GB is the amount of memory on the node.
If you run serial jobs you must still use all 40 cores on the node.  Visit the serial jobs page for examples of how to do this.
Since there are 40 cores per node, your job should use N x 40 cores. If you do not, we will contact you to help you optimize your workflow.  Or you can contact us to get assistance.

== Limits ==

There are limits to the size and duration of your jobs, the number of jobs you can run and the number of jobs you can have queued.  It matters whether a user is part of a group with a Resources for Research Group allocation or not. It also matters in which 'partition' the jobs runs. 'Partitions' are SLURM-speak for use cases.  You specify the partition with the -p parameter to sbatch or salloc, but if you do not specify one, your job will run in the compute partition, which is the most common case.

Usage                                                   	Partition   	Running jobs            	Submitted jobs (incl. running)	Min. size of jobs	Max. size of jobs       	Min. walltime	Max. walltime
Compute jobs with an allocation                         	compute     	50                      	1000                          	1 node (40 cores)	1000 nodes (40000 cores)	15 minutes   	24 hours
Compute jobs without allocation ("default")             	compute     	50                      	200                           	1 node (40 cores)	20 nodes (800 cores)    	15 minutes   	24 hours
Testing or troubleshooting                              	debug       	1                       	1                             	1 node (40 cores)	4 nodes (160 cores)     	N/A          	1 hour
Archiving or retrieving data in HPSS                    	archivelong 	2 per user (max 5 total)	10 per user                   	N/A              	N/A                     	15 minutes   	72 hours
Inspecting archived data, small archival actions in HPSS	archiveshort	2 per user              	10 per user                   	N/A              	N/A                     	15 minutes   	1 hour

Within these limits, jobs will still have to wait in the queue.  The waiting time depends on many factors such as the allocation amount, how much allocation was used in the recent past, the number of nodes and the walltime, and how many other jobs are waiting in the queue.

== File Input/Output Tips ==

It is important to understand the file systems, so as to perform your file I/O (Input/Output) responsibly.  Refer to the  Data management at Niagara page for details about the file systems.
* Your files can be seen on all Niagara login and compute nodes.
* $HOME, $SCRATCH, and $PROJECT all use the parallel file system called GPFS.
* GPFS is a high-performance file system which provides rapid reads and writes to large data sets in parallel from many nodes.
* Accessing data sets which consist of many, small files leads to poor performance on GPFS.
* Avoid reading and writing lots of small amounts of data to disk.  Many small files on the system waste space and are slower to access, read and write.  If you must write many small files, use ramdisk.
* Write data out in a binary format. This is faster and takes less space.
* The Burst Buffer is better for i/o heavy jobs and to speed up checkpoints.

== Example submission script (MPI) ==

#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=80
#SBATCH --time=1:00:00
#SBATCH --job-name mpi_job
#SBATCH --output=mpi_output_%j.txt
#SBATCH --mail-type=FAIL

cd $SLURM_SUBMIT_DIR

module load intel/2018.2
module load openmpi/3.1.0

mpirun ./mpi_example
# or "srun ./mpi_example"

Submit this script from your scratch directory with the command:
nia-login07:scratch$ sbatch mpi_job.sh

First line indicates that this is a bash script.
Lines starting with #SBATCH go to SLURM.
sbatch reads these lines as a job request (which it gives the name mpi_job)
In this case, SLURM looks for 2 nodes (each of which will have 40 cores) on which to run a total of 80 tasks, for 1 hour.(Instead of specifying --ntasks=80, you can also ask for --ntasks-per-node=40, which amounts to the same.)
Note that the mpifun flag "--ppn" (processors per node) is ignored.
Once it found such a node, it runs the script:

Change to the submission directory;
Loads modules;
Runs the mpi_example application (SLURM will inform mpirun or srun on how many processes to run).

To use hyperthreading, just change --ntasks=80 to --ntasks=160, and add --bind-to none to the mpirun command (the latter is necessary for OpenMPI only, not when using IntelMPI).

== Example submission script (OpenMP) ==

#!/bin/bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=1:00:00
#SBATCH --job-name openmp_job
#SBATCH --output=openmp_output_%j.txt
#SBATCH --mail-type=FAIL

cd $SLURM_SUBMIT_DIR

module load intel/2018.2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./openmp_example
# or "srun ./openmp_example".

Submit this script from your scratch directory with the command:
nia-login07:scratch$ sbatch openmp_job.sh
* First line indicates that this is a bash script.
* Lines starting with #SBATCH go to SLURM.
* sbatch reads these lines as a job request (which it gives the name openmp_ex) .
* In this case, SLURM looks for one node with 40 cores to be run inside one task, for 1 hour.
* Once it found such a node, it runs the script:
** Change to the submission directory;
** Loads modules (must be done again in the submission script on Niagara);
** Sets an environment variable to set the number of threads to 40 (no hyperthreading in this example);
** Runs the appl_openmp_ex application.
* To use hyperthreading, just change --cpus-per-task=40 to --cpus-per-task=80.

== Monitoring queued jobs ==

Once the job is incorporated into the queue, there are some command you can use to monitor its progress.

squeue or sqc (a caching version of squeue) to show the job queue (squeue -u $USER for just your jobs);
qsum shows a summary of qudue by user
squeue -j JOBID to get information on a specific job
(alternatively, scontrol show job JOBID, which is more verbose).
squeue --start -j JOBID to get an estimate for when a job will run; these tend not to be very accurate predictions.
scancel -i JOBID to cancel the job.
jobperf JOBID to get an instantaneous view of the cpu and memory usage of the nodes of the job while it is running.
sacct to get information on your recent jobs.

Further instructions for monitoring your jobs can be found on the Slurm page.  The my.SciNet site is also a very useful tool for monitoring your current and past usage.

= Visualization =
Information about how to use visualization tools on Niagara is available on Visualization page.

= Further information =

Useful sites

* SciNet: https://www.scinet.utoronto.ca
* Niagara: Niagara wiki page
* System Status: https://docs.scinet.utoronto.ca/index.php/Main_Page
* Training: https://support.scinet.utoronto.ca/education

Support
Contact our Technical support