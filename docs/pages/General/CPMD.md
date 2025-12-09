---
title: "CPMD/en"
url: "https://docs.alliancecan.ca/wiki/CPMD/en"
category: "General"
last_modified: "2024-04-30T21:32:07Z"
page_id: 11471
display_title: "CPMD"
---

`<languages />`{=html}

[CPMD](https://www.cpmd.org/wordpress/) is a plane wave/pseudo-potential DFT code for ab initio molecular dynamics simulations.

# License limitations {#license_limitations}

In the past, access to CPMD required registration and confirmation with the developers, but registration on their website is no longer needed. However, the modules installed on our clusters are still protected by a POSIX group.

Before you can start using [CPMD](http://cpmd.org) on our clusters, [ send us a support request](https://docs.alliancecan.ca/Technical_support " send us a support request"){.wikilink} and ask to be added to the POSIX group that will allow you to access the software.

# Module

You can access CPMD by loading a [module](https://docs.alliancecan.ca/Utiliser_des_modules/en "module"){.wikilink}.

``` bash
module load StdEnv/2020
module load intel/2020.1.217 openmpi/4.0.3 cpmd/4.3
```

# Local installation of CPMD {#local_installation_of_cpmd}

It has recently been our experience that a response from CPMD admins can unfortunately take weeks or even months. If you are a registered CPMD user, you have access to the CPMD source files and can therefore build the software yourself in your /home directory using our software environment called EasyBuild, with the exact same recipe that we would use for a central installation.

Below are instructions on how to build CPMD 4.3 under your account on the cluster of your choice:

Create a local directory like so

`$ mkdir -p ~/.local/easybuild/sources/c/CPMD`

Place all the CPMD source tarballs and patches into that directory.

    $ ls -al ~/.local/easybuild/sources/c/CPMD
    cpmd2cube.tar.gz
    cpmd2xyz-scripts.tar.gz
    cpmd-v4.3.tar.gz
    fourier.tar.gz
    patch.to.4612
    patch.to.4615
    patch.to.4616
    patch.to.4621
    patch.to.4624
    patch.to.4627

Then run the EasyBuild command.

`$ eb CPMD-4.3-iomkl-2020a.eb --rebuild`

The `--rebuild` option forces EasyBuild to ignore CPMD 4.3 installed in a central location and proceed instead with the installation in your /home directory.

Once the software is installed, log out and log back in.

Now, when you type `module load cpmd`, the software installed in your /home directory will get picked up.

    $ module load StdEnv/2020
    $ module load intel/2020.1.217 openmpi/4.0.3 cpmd/4.3
    $ which cpmd.x
    ~/.local/easybuild/software/2020/avx2/MPI/intel2020/openmpi4/cpmd/4.3/bin/cpmd.x

You can use it now as usual in your submission script.

# Example of a job script {#example_of_a_job_script}

To run a job, you will need to set an input file and access to the pseudo-potentials.

If the input file and the pseudo-potentials are in the same directory, the command to run the program in parallel is:

`<code>`{=html}srun cpmd.x `<input files>`{=html} \>

<output file>

`</code>`{=html} (as in the script 1)

It is also possible to put the pseudo-potentials in another directory with

`<code>`{=html}srun cpmd.x `<input files>`{=html} `<path to pseudo potentials location>`{=html} \>

<output file>

`</code>`{=html} (as in script 2)

`<tabs>`{=html} `<tab name="INPUT">`{=html}

`</tab>`{=html}

`<tab name="Script 1">`{=html} ``{{File
  |name=run-cpmd.sh
  |lang="bash"
  |contents=
#!/bin/bash

#SBATCH --account=def-someacct
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2500M
#SBATCH --time=0-1:00

# Load the modules:

module load StdEnv/2020
module load intel/2020.1.217 openmpi/4.0.3 cpmd/4.3

echo "Starting run at: `date`"

CPMD_INPUT="1-h2-wave.inp"
CPMD_OUTPUT="1-h2-wave_output.txt"

srun cpmd.x ${CPMD_INPUT} > ${CPMD_OUTPUT}

echo "Program finished with exit code $? at: `date`"
}}``{=mediawiki} `</tab>`{=html}

`<tab name="Script 2">`{=html} ``{{File
  |name=run-cpmd.sh
  |lang="bash"
  |contents=
#!/bin/bash

#SBATCH --account=def-someacct
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2500M
#SBATCH --time=0-1:00

# Load the modules:

module load StdEnv/2020
module load intel/2020.1.217 openmpi/4.0.3 cpmd/4.3

echo "Starting run at: `date`"

CPMD_INPUT="1-h2-wave.inp"
CPMD_OUTPUT="1-h2-wave_output.txt"
PP_PATH=<path to the location of pseudo-potentials>

srun cpmd.x ${CPMD_INPUT} ${PP_PATH} > ${CPMD_OUTPUT}

echo "Program finished with exit code $? at: `date`"
}}``{=mediawiki} `</tab>`{=html} `</tabs>`{=html}

# Related link {#related_link}

- CPMD [home page](https://www.cpmd.org/wordpress/).
