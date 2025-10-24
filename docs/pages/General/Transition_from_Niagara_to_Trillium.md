---
title: "Transition from Niagara to Trillium"
url: "https://docs.alliancecan.ca/wiki/Transition_from_Niagara_to_Trillium"
category: "General"
last_modified: "2025-10-22T16:54:21Z"
page_id: 29084
display_title: "Transition from Niagara to Trillium"
---

`<languages />`{=html}

`<translate>`{=html}

## Getting Access {#getting_access}

### Opt-in {#opt_in}

For Niagara and Mist, users had to opt-in on CCDB to use the system. The same is true for the Trillium cluster, but the webpage on which you can do this has changed to [<https://ccdb.alliancecan.ca/me/access_systems>](https://ccdb.alliancecan.ca/me/access_systems) (on which you can also opt in to other national services).

Users that had opted into Niagara/Mist on August 5, 2025 will have automatically been opted-in to Trillium.

Opting into Trillium gives you access to both the CPU and GPU subclusters of Trillium.

To get acccess to HPSS, the nearline system for Trillium, there is a separate opt-in on [<https://ccdb.alliancecan.ca/me/access_systems>](https://ccdb.alliancecan.ca/me/access_systems); here too, you if you had access to HPSS through Niagara before August 5th, you will be opted-in automatically. For the moment, HPSS has not been integrated into Trillium yet. Once it has, the files on HPSS will not have changed as it is still the same storage system.

### Logging in {#logging_in}

#### Terminal access {#terminal_access}

You can login to the CPU subcluster of Trillium by ssh-ing to trillium.alliancecan.ca:

    $ ssh USERNAME@trillium.alliancecan.ca

The first time you login, please make sure you are actually accessing Trillium by checking if the [login node ssh host key fingerprint](https://docs.alliancecan.ca/SSH_security_improvements/en#Trillium "login node ssh host key fingerprint"){.wikilink} matches. As was the case on Niagara and Mist, you will need to use [SSH Keys](https://docs.alliancecan.ca/SSH_Keys "SSH Keys"){.wikilink} and have MFA enabled on your CCDB account. This will get you onto one of six CPU login nodes called tri-login01-6. These login nodes do not have GPUs and *can only submit jobs to CPU compute nodes*.

To access the GPU subcluster of Trillium, you should log into trillium-gpu.alliancecan.ca (also with ssh keys and MFA):

    $ ssh USERNAME@trillium-gpu.alliancecan.ca

This will land you on the GPU login node trig-login01, which has 4 NVIDIA H100 GPU, and from which you can only *submit to the compute nodes of the GPU subcluster*.

#### First time actions {#first_time_actions}

After logging in for the first time, you will see the files that were on Niagara. Your initialization files likely won\'t work, or not exactly as they should. We **\'highly recommend** that you run the following command:

    $ trisetup

which will populate the following files in your home directory: .bashrc, .bash_profile, .chsrc. and .Xauthority. It will also create the directories .licenses, .local, .ssh, and links. The latter contains symbolic links to your scratch and project directories. Your original versions of these files will be saved under a different name with a date stamp.

After this, you can recompile your code, reinstall virtual environments, etc. (see below).

#### Web access {#web_access}

The [SciNet OnDemand website](https://ondemand.scinet.utoronto.ca) will remain connected to Niagara for now, but will eventually also offer a way to log into Trillium and run web-based apps like Jupyter with access to the Trillium file system.

## Hardware changes {#hardware_changes}

Trillium consists of two homogeneous subclusters, a CPU subcluster and a GPU subcluster.

### CPUs

Each compute node of the CPU subcluster has 192 cores (Niagara had 40) and 755 GB of available memory (Niagara had 188 GB). The CPUs are AMD Zen 5 chips - a.k.a Turin (Niagara had Intel Skylake and Cascaselake chips). There are 1224 compute nodes in the Trillium CPU subcluster, for a total of 235,008 cores.

If you were compiling code that used math and linear algebra routines from the Intel MKL, we suggest you switch to [ Flexiblas ](https://docs.alliancecan.ca/BLAS_and_LAPACK " Flexiblas "){.wikilink}, or use the AMD AOCL libraries directly; these are in available in the `aocl-blas` and `aocl-lapack` modules.

### GPUS

Each compute nodes in the GPU subcluster has 96 cores and 755 GB of available memory, and 4 GPUs. The CPUs are AMD Zen 4 (a.k.a. Genoa) chips, while the GPUs are NVIDIA H100 (Mist\'s GPUs were V100). There are 61 GPU compute nodes, so in total Trillium has 244 GPUs.

It is important to make sure that your jobs are not wasting resources. They should either use all of the cores, or exhaust most of the memory, or efficiently use GPUs.

## Storage

### Automatic data migration {#automatic_data_migration}

Any data that was present on your home, scratch, and project directories on Niagara on July 31st and that you did not change after that, will have been copied over to the corresponding directory on Trillium. Any data that was added or modified afterwards on Niagara may need to be copied over by the user. For example, a user might ssh to Niagara datamover 1:

    $ ssh USERNAME@nia-datamover1.scinet.utoronto.ca

and take advantage that both Niagara and Trillium file systems are mounted there to copy locally from one to the other:

    USERNAME@nia-dm1:~$ cp /home/G/GROUP/USERNAME/file.txt /trillium_home/USERNAME/file.txt

For now, the quotas of home and scratch remain the same default values as they were on Niagara, i.e., 100 GB and 25 TB, respectively.

### Directory locations {#directory_locations}

The directory layout of home, scratch, and project has changed compared to that on Niagara. Your home and scratch directories will not be nested under a group, but will be at **/home/USERNAME** and **/scratch/USERNAME**.

The scratch quota and purging settings will be reviewed in the coming months.

If your group had a project directory on Niagara, you will now find that at **/project/RRG-NAME**, where RRG-NAME is the name of your rrg allocation. In addition all groups now have a project directory **/project/def-PINAME** with a quota of 1 TB.

As before, home and project will be backed up, but scratch will not.

To make it easier to find these locations, a directory called **links** can be created inside your \$HOME contains with links to your scratch and project directories. To get this set of links, please run the following command, once, after logging in:

    $ trisetup

This will also update your .bashrc and other startup files; the command will show exactly what it did and where it saves any earlier version.

Once setup, the links in `$HOME/links/project` will be automatically updated if you join, leave, or move groups.

### File and directory permissions {#file_and_directory_permissions}

Your home directory on Niagara had you as the owner but your PI\'s group as the group-ownership. In contrast, on Trillium, your files and directories in home have the group-ownership of your own, private group. This means that sharing files on home with other group members will require you to set the permissions using ACLs or change the group ownership of files.

The situation is similar for scratch, in that new users will see their scratch directory to be private, i.e., it has the group-ownership set to their private group; if you want to share files with someone else, you will need to use ACL or change group ownerships. It should be noted, however, that users whose files were transfered from Niagara, will see those having the group-ownership of the research group under which their scratch was originally nested.

The project directories are setup for sharing and as such have group-membership corresponding to their def or rrg allocation.

The home and project file systems are read-only on compute nodes on Trillium. Your compute jobs, therefore, have to write to the scratch directory. Copying data from scratch to project or home should be restricted to data that really needs to be conserved and backed up. For this, you can either use the scp or rsync commands to a datamover or login node at the end of your job, or log into the datamover node and perform the transfer there.

## Software

### Software stack {#software_stack}

The default NiaEnv software stack that was available on Niagara will be retired and will not be available on Trillium. Likewise, the MistEnv software stack on Mist will not be available on Trillium-gpu. The CVMFS software stack that is available on the other national clusters will be used on Trillium as well. However, only the **gentoo/2023** and **CCconfig** modules will be loaded by default; all other modules (for compilers, libraries, python, etc) will have to be explicitly loaded.

To get the same set of default modules as on the other national clusters, you should load the StdEnv/2023 module:

    $ module load StdEnv/2023

This will load the gcc compiler as the default compiler, as well as many other modules.

To compile GPU code with CUDA, you also need to load the cuda module:

    trig-login01$ module load StdEnv/2023 cuda/12.6

The cuda module is not available on the Trillium CPU nodes, it only works the its GPU login node and GPU compute nodes.

### Recompilation required {#recompilation_required}

Even though all your files, including binaries, were copied over, codes that weren't using CVMFS software stack, or that used absolute paths, will have to be recompiled. In fact, it is recommended to all your code, even if they did use the CVMFS stack on Niagara.

### Recreate virtual python environments required {#recreate_virtual_python_environments_required}

Even though all your files were copied over, your virtual environments will not work. You must recreate them because of the users' directory change.

If you had anaconda virtual environments, you will find that there is no anaconda module. You should switch to using virtual environments instead.

## Training, documentation, and support {#training_documentation_and_support}

- New quickstart: [Trillium Quickstart](https://docs.alliancecan.ca/Trillium_Quickstart "Trillium Quickstart"){.wikilink}
- Support email: <trillium@tech.alliancecan.ca>
- Self-guided \"[Intro to SciNet and Trillium](https://education.scinet.utoronto.ca/course/view.php?id=1389)\"

`</translate>`{=html}
