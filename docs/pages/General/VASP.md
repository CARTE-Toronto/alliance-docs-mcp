---
title: "VASP/en"
url: "https://docs.alliancecan.ca/wiki/VASP/en"
category: "General"
last_modified: "2025-11-18T19:41:29Z"
page_id: 4649
display_title: "VASP"
---

<languages />
:<i>The Vienna ab initio Simulation Package (VASP) is a computer program for atomic scale materials modelling, e.g. electronic structure calculations and quantum mechanical molecular dynamics, from first principles.</i> 
: Reference: [VASP website](https://www.vasp.at/)

## Licensing
VASP is commercial software which can only be licensed to groups that are hired by a single legal entity, which is incompatible with the way we operate. We have tried to negotiate an agreement with the licensor which would let us install the software everywhere on our infrastructure, but without success. For that reason you may have to install VASP yourself (though see <i>Site exceptions</i> below). Please read the terms of your own license, as you are likely subject to the same restriction. This limits the support we are allowed to offer to users who need help installing the software. Please see [VASP yourself</i>](https://docs.alliancecan.ca/<i>Building)(#Building VASP yourself) below for instructions on installing the software. 

### Site exceptions
Simon Fraser University, the University of Waterloo, and the University of Toronto own [Fir](https://docs.alliancecan.ca/Fir), [Nibi](https://docs.alliancecan.ca/Nibi), and [Trillium](https://docs.alliancecan.ca/Trillium), respectively, and have a license with VASP. Some of their employees are therefore allowed to access and install specific versions of VASP on those clusters and provide limited support. 

If you wish to use the prebuilt VASP binaries on [Fir](https://docs.alliancecan.ca/Fir), [Nibi](https://docs.alliancecan.ca/Nibi) and/or [Trillium](https://docs.alliancecan.ca/Trillium), you must contact [support](https://docs.alliancecan.ca/Technical)(Technical support) requesting access to VASP with the following information: 
- Include license holder (your PI) information:
** Name 
** Email address
** Department and institution (university) 
- Include license information:
** Version of the VASP license (<b>VASP version 4 or version 5</b>)
** <b>License number</b>
** Provide an updated list of who is allowed to use your VASP license. For example, forward to us the most recent email from the VASP license administrator that contains the list of licensed users.

If you are licensed for version 5 you may also use version 4, but a version 4 license does not permit you to use version 5. The same for version 6, if you are licensed for version 6 you may also use versions 5 and 4.

## Using prebuilt VASP

Prebuilt VASP binary files have been installed only on Fir, Nibi, and Trillium (replacing Cedar, Graham, and Niagara). According to our license agreement, we are not allowed to install VASP centrally on other computer clusters. However,  users may want to install VASP on their own home directory. Please have a look at [VASP yourself</i>](https://docs.alliancecan.ca/<i>Building)(VASP#Building VASP yourself). To load prebuilt VASP on Fir and Nibi, please do the following: 

For vasp/5.4.4 
 module load StdEnv/2023  intel/2023.2.1  openmpi/4.1.5
 module load vasp/5.4.4
For vasp/6.4.2
 module load StdEnv/2023 intel/2023.2.1 intelmpi/2021.9.0
 module load vasp/6.4.2

1. Run `module spider vasp` to see which versions are available.
1. Choose your version and run `module spider vasp/<version>` to see which dependencies you need to load for this particular version.
1. Load the dependencies and the VASP module, for example:
 module load intel/2020.1.217  intelmpi/2019.7.217 vasp/5.4.4
See [modules](https://docs.alliancecan.ca/Using)(Using modules) for more information.

To use VASP on Trillium, modules may be loaded as follows:

For vasp/5.4.4
 module load StdEnv/2023 intel/2023.2.1 intelmpi/2021.9.0 
 module load imkl/2023.2.0 
 module use /opt/software/commercial/modules 
 module load vasp/5.4.4

For vasp/6.4.2
 module load StdEnv/2023 intel/2023.2.1 intelmpi/2021.9.0 hdf5/1.14.2
 module use /opt/software/commercial/modules
 module load vasp/6.4.2
For general usage of Trillium, please see the [Quickstart](https://docs.alliancecan.ca/Trillium)(Trillium Quickstart) page.

### Pseudopotential files
All pseudopotentials have been downloaded from the official VASP website and untarred. They are all located in ` $EBROOTVASP/pseudopotentials/` on Cedar and Graham and can be accessed once the VASP module is loaded.

### Executable programs

<b>For VASP-4.6</b>, executable files are: 
- `vasp` for standard NVT calculations with non gamma k points 
- `vasp-gamma` for standard NVT calculations with only gamma points 
- `makeparam` to estimate how much memory is required to run VASP for a particular cluster

<b>For VASP-5.4.1, 5.4.4 and 6.1.0 (without CUDA support)</b>, executable files are:
- `vasp_std` for standard NVT calculations with non gamma k points
- `vasp_gam` for standard NVT calculations with only gamma points
- `vasp_ncl` for NPT calculations with non gamma k points

<b>For VASP-5.4.4 and 6.1.0 (with CUDA support)</b>, executable files are:
- `vasp_gpu` for standard NVT calculations with gamma and non gamma k points
- `vasp_gpu_ncl` for NPT calculations with gamma and non gamma k points

Two extensions have also been incorporated:
- [Transition State Tools](http://theory.cm.utexas.edu/vtsttools/)
- [VASPsol](https://github.com/henniggroup/VASPsol)

If you need a version of VASP that does not appear here, you can either build it yourself (see below) or [ write to us](https://docs.alliancecan.ca/Technical support ) and ask that it be built and installed.

## Vasp-GPU
[Fig.1 Simulation time as a function of the number of CPUs for GPU=0, 1, and 2](https://docs.alliancecan.ca/400px|thumb|)(File:VASP benchmark ENG.png)
Vasp-GPU executable files run on both GPUs and CPUs of a node. Basically, calculation on a GPU is much more expensive than on a CPU, therefore we highly recommend to perform a benchmark using one or 2 GPUs to make sure they are getting a maximum performance from the GPU use. Fig.1 shows a benchmark of Si crystal which contains 256 Si-atoms in the simulation box. Blue, black and red lines show simulation time as a function of Number of CPU for GPU=0, 1, and 2 respectively. It shows the performance for GPU=1,2 and CPU=1 is more than 5 times better compared to GPU=0 and CPU=1. However, a comparison of calculations with GPU=1 and GPU=2 indicates that there is not much performance gain from GPU=1 to GPU=2. In fact, use for GPU=2 is around 50% in our monitoring system. Therefore we recommend users to first perform a benchmark like this for their own system to make sure they are not wasting any computer resources.

## Example of a VASP job script

The following is a job script to run VASP in parallel using the Slurm job scheduler:

*The above job script requests four CPU cores and 4096MB memory (4x1024MB).
*<ACCOUNT> is a Slurm account name; see [and projects](https://docs.alliancecan.ca/Accounts)(Running_jobs#Accounts_and_projects) to know what to enter there.
*<VERSION> is the number for the VASP version you want to use: 4.6, 5.4.1, 5.4.4 or 6.1.0.
*Use module spider vasp/<VERSION> to see how you can change this particular version.
*<VASP> is the name of the executable. Refer to section *Executable programs* above for the executables you can select for each version. 

*The above job script requests one CPU core and 1024MB memory.
*The above job script requests one GPU type p100 which is only available on Cedar. For other clusters, please see the [types available](https://docs.alliancecan.ca/GPU)(Using GPUs with Slurm#Available_hardware). 
*The above job uses `srun` to run VASP.

VASP uses four input files named as INCAR, KPOINTS, POSCAR, POTCAR. It is best to prepare VASP input files in a separate directory for each job. To submit the job from that directory, use:
 sbatch vasp_job.sh

If you do not know how much memory you need for your job, prepare all your input files and then run `makeparam` in an [job submission](https://docs.alliancecan.ca/interactive)(Running_jobs#Interactive_jobs). Then use the result as required memory for the next run. However, for a more accurate estimate for future jobs, check the maximum stack size used by [jobs](https://docs.alliancecan.ca/completed)(Running_jobs#Completed_jobs) and use this as the memory requirement per processor for the next job.

If you want to use 32 or more cores, please read about [scheduling](https://docs.alliancecan.ca/whole-node)(Job_scheduling_policies#Whole_nodes_versus_cores).

## Building VASP yourself

If you are licensed to use VASP and have access to VASP source code, you can install various versions of VASP in your /home directory on all our clusters using the following [EasyBuild](https://docs.alliancecan.ca/EasyBuild) commands. 

` eb -f [NAME](https://docs.alliancecan.ca/RECIPE) --sourcepath=[SOURCEPATH] `

where [SOURCEPATH] is the directory containing the VASP source code and [NAME](https://docs.alliancecan.ca/RECIPE) is the name of the recipe. The first tab of table bellow shows the list of available recipes along with the corresponding required source files. In this table VTSTtools and  vaspSOL are Transition State Tools and VASPsol extensions respectively. Second tabs of this table shows the list of the libraries that are included to vasp. You may download the source code from the [VASP website](https://www.vasp.at/). Running the command will take some time, perhaps more than an hour. Once it is done, you will be able to load and run VASP using `module` commands just as explained above in [prebuilt VASP](https://docs.alliancecan.ca/Using)(VASP#Using_prebuilt_VASP).

Alternatively to build a custom version of VASP, please see [software in your home directory](https://docs.alliancecan.ca/Installing)(Installing software in your home directory) and 
[Installing VASP 5](https://www.vasp.at/wiki/index.php/Installing_VASP.5.X.X) or [Installing VASP 6](https://www.vasp.at/wiki/index.php/Installing_VASP.6.X.X).

<tabs>
<tab name="Recipe specification and implementation">
{| class="wikitable"
|-
! Recipe Name
! Version
! Environment
! Source file
! CPU/GPU
! VTSTtools
! vaspSOL
|-
| VASP-5.4.4-iimpi-2020a.eb
| 5.4.4
| StdEnv/2023
| vasp.5.4.4.pl2.tgz
| CPU
| yes
| yes
|-
| VASP-6.1.2-iimpi-2020a.eb
| 6.1.2
| StdEnv/2020
| vasp.6.1.2_patched.tgz
| CPU
| yes
| yes
|-
| VASP-6.2.1-iimpi-2020a.eb
| 6.2.1
| StdEnv/2020
| vasp.6.2.1.tgz
| CPU
| yes
| yes
|-
| VASP-6.3.0-iimpi-2020a.eb
| 6.3.0
| StdEnv/2020
| vasp.6.3.0.tgz
| CPU
| yes
| yes
|-
| VASP-6.3.1-iimpi-2020a.eb
| 6.3.1
| StdEnv/2020
| vasp.6.3.1.tgz
| CPU
| yes
| yes
|-
| VASP-6.4.2-iimpi-2023a.eb
| 6.4.2
| StdEnv/2023
| vasp.6.4.2.tar
| CPU
| yes
| yes
|-
| VASP-6.4.3-iimpi-2023a.eb
| 6.4.3
| StdEnv/2023
| vasp.6.4.3.tar
| CPU
| yes
| yes
|-
| VASP-6.5.0-iimpi-2023a.eb
| 6.5.0
| StdEnv/2023
| vasp.6.5.0.tgz
| CPU
| No
| No
|-
| VASP-6.5.1-iimpi-2023a.eb
| 6.5.1
| StdEnv/2023
| vasp.6.5.1.tgz
| CPU
| No
| No
|-  
|}
</tab>
<tab name="Included Libraries">
{| class="wikitable"
|-
! Recipe Name
! Wannier Function
! Beef
! HDF5
! LibXC
! ELPA
! Libmbd
! dft4
|-
| VASP-5.4.4-iimpi-2020a.eb
| Yes
| Yes
| No
| No
| No
| No
| No
|-
| VASP-6.1.2-iimpi-2020a.eb
| Yes
| Yes
| No
| No
| No
| No
| No
|-
| VASP-6.2.1-iimpi-2020a.eb
| Yes
| Yes
| No
| No
| No
| No
| No
|- 
| VASP-6.3.0-iimpi-2020a.eb
| Yes
| Yes
| Yes
| Yes
| No
| No
| No
|-
| VASP-6.3.1-iimpi-2020a.eb
| Yes
| Yes
| Yes
| Yes
| No
| No
| No
|-
| VASP-6.4.2-iimpi-2023a.eb
| Yes
| Yes
| Yes
| Yes
| No
| No
| No
|-
| VASP-6.4.3-iimpi-2023a.eb
| Yes
| Yes
| Yes
| Yes
| No
| No
| Yes
|-
| VASP-6.5.0-iimpi-2023a.eb
| Yes 
| Yes
| Yes
| Yes
| Yes
| Yes
| Yes
|-
| VASP-6.5.1-iimpi-2023a.eb
| Yes 
| Yes
| Yes
| Yes
| Yes
| Yes
| Yes
|-
</tab>
</tabs>

1. External links

- [Getting Started](https://www.vasp.at/tutorials/latest/part1/) guide from the developers' Web site.
- [py4vasp](https://www.vasp.at/py4vasp/latest/) is a Python interface to extract data from VASP calculations.