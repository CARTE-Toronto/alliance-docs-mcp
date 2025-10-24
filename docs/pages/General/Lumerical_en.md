---
title: "Lumerical/en"
url: "https://docs.alliancecan.ca/wiki/Lumerical/en"
category: "General"
last_modified: "2022-12-12T15:34:23Z"
page_id: 6301
display_title: "Lumerical"
---

`<languages />`{=html}

[Lumerical](https://www.lumerical.com/) is a suite of applications for modelling [nanophotonic](https://en.wikipedia.org/wiki/Nanophotonics) devices, which includes [FDTD Solutions](https://www.lumerical.com/tcad-products/fdtd/).

# Installation

FDTD Solutions is now available as part of the Lumerical package. Compute Canada does not have a central installation of the Lumerical suite or FDTD Solutions. However, if you are licensed to use the software, you can install it following the instructions below.

If you have downloaded whole Lumerical suite (e.g. filename: `Lumerical-2020a-r1-d316eeda68.tar.gz`), follow the instructions in sections \"Installing Lumerical\" and \"Using the Lumerical module\". If you have downloaded FDTD Solutions on it\'s own (e.g. filename: `FDTD_Solutions-8.19.1438.tar.gz`), follow the instructions in sections \"Installing FDTD Solutions\" and \"Using the fdtd_solutions module\".

## Installing Lumerical {#installing_lumerical}

### In case the installer release matches that of the recipe {#in_case_the_installer_release_matches_that_of_the_recipe}

To install the Lumerical suite run the command `<path>`{=html} \--disable-enforce-checksums}} where `path` is the path to the folder containing the `.tar.gz` file to install Lumerical on Linux.

### In case the installer release does not match that of the recipe {#in_case_the_installer_release_does_not_match_that_of_the_recipe}

With a different 2020a release than 2020a-r1-d316eeda68, run `<version>`{=html} \--sourcepath`<path>`{=html} \--disable-enforce-checksums}} For example, if `Lumerical-2020a-r1-d316eeda68.eb.tar.gz` is downloaded in `$HOME/scratch`, the following command will install Lumerical within your `$HOME/.local` folder. 2020a-r6-aabbccdd \--sourcepath\$HOME/scratch \--disable-enforce-checksums}}

It is important that the version of the installation recipe (year plus \'a\' or \'b\') needs to exactly match that of the installer. If either the letter or the year changes (e.g. from 2020a to 2020b), we will need to adapt the installation script to the new version.

As of April 1st, 2020 we have the following installation recipes available:

  Installation recipe                        Intended for Installer                         Compatible with Installers
  ------------------------------------------ ---------------------------------------------- ----------------------------
  `Lumerical-2019b-r6-1db3676.eb`            `Lumerical-2019b-r6-1db3676.tar.gz`            `Lumerical-2019b-*.tar.gz`
  `Lumerical-2020a-r1-d316eeda68.eb`         `Lumerical-2020a-r1-d316eeda68.tar.gz`         `Lumerical-2020a-*.tar.gz`
  `Lumerical-2021-R2.5-2885-27742aa972.eb`   `Lumerical-2021-R2.5-2885-27742aa972.tar.gz`   `Lumerical-2021-*.tar.gz`
  `Lumerical-2022-R1.3-3016-2c0580a.eb`      `Lumerical-2022-R1.3-3016-2c0580a.tar.gz`      `Lumerical-2022-*.tar.gz`

If this does not work, please contact our [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink} and we will adapt an installation recipe for your version.

Once installed, you will need to log out and back into the server. To load the Lumerical module, use

### Configuring your own license file {#configuring_your_own_license_file}

The Lumerical module will look for the file `$HOME/.licenses/lumerical.lic` to determine how to contact the license server. Create the file with the following content, adjusting `27011@license01.example.com` to the port and hostname of your license server.

copy the content below to `$HOME/.licenses/lumerical.lic`

`setenv("LUMERICAL_LICENSE_FILE", "27011@license01.example.com")`

## Installing FDTD Solutions {#installing_fdtd_solutions}

To install FDTD Solutions, run the command `<path>`{=html} \--disable-enforce-checksums}} where `path` is the path to the folder containing the `.tar.gz` file to install FDTD Solutions on Linux.

With a version other than 8.19.1438, run `<version>`{=html} \--sourcepath`<path>`{=html} \--disable-enforce-checksums}} For example, if `FDTD_Solutions-8.19.1466.tar` is downloaded in `$HOME/Downloads`, the following command will install FDTD Solution within your `$HOME/.local` folder. 8.19.1466 \--sourcepath\$HOME/Downloads \--disable-enforce-checksums}}

If this does not work, please contact our [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink} and we will adapt an installation script for your version.

Once installed, you will need to log out and back into the server. To load the FDTD module, use

You will also need to set up your installation to use your license server. Start the software first on a login node; it should ask you for information about the license server. You will only need to do this once.

# Using the software {#using_the_software}

The main difference between the modules `fdtd_solutions` and `lumerical`, beside the fact that the Lumerical module contains additional tools, is that the environment variable that contains the install location is named `EBROOTFDTD_SOLUTIONS` and `EBROOTLUMERICAL` respectively. This means scripts written for one module should be adjusted for the other by replacing the name of the module in the `module load ...` line and replacing `EBROOTFDTD_SOLUTIONS` with `EBROOTLUMERICAL` or vice versa.

## Using the Lumerical module {#using_the_lumerical_module}

The MPI implementation provided by Lumerical is not tightly coupled with our scheduler. Because of this, you should use options `--ntasks-per-node=1` and `--cpus-per-task=32` when submitting a job.

Your submission script should look like the following example, where two nodes are requested for 30 minutes. You can adjust the time limit and the node count to fit your needs. `{{File  
  |name=lumrical_job.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --time=0:30:00            # time limit (D-HH:MM:SS)
#SBATCH --ntasks-per-node=1       # do not change this number
#SBATCH --cpus-per-task=32        # adjust to number of cores per node
#SBATCH --ntasks=2       # the same number as nodes, one task per node
#SBATCH --nodes=2
#SBATCH --mem=0          # special value, requests all memory on node
module load lumerical

MPI=$EBROOTLUMERICAL/mpich2/nemesis/bin/mpiexec
MY_PROG=$EBROOTLUMERICAL/bin/fdtd-engine-mpich2nem

INPUT="avalanche_photodetector_optical.fsp"
NCORE=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))

$MPI -n $NCORE $MY_PROG ./${INPUT}

}}`{=mediawiki}

## Using the fdtd_solutions module {#using_the_fdtd_solutions_module}

The MPI implementation provided by FDTD is not tightly coupled with our scheduler. Because of this, you should use options `--ntasks-per-node=1` and `--cpus-per-task=32` when submitting a job.

Your submission script should look like the following example, where two nodes are requested for one hour. You can adjust the time limit and the node count to fit your needs.

```{=mediawiki}
{{File  
  |name=fdtd_solutions.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --time=0:30:00            # time limit (D-HH:MM:SS)
#SBATCH --ntasks-per-node=1    # do not change this number
#SBATCH --cpus-per-task=32     # do not change this number
#SBATCH --ntasks=2    # the same number as nodes, one task per node
#SBATCH --nodes=2
#SBATCH --mem=0       # special value, requests all memory on node
module load fdtd_solutions
MPI=$EBROOTFDTD_SOLUTIONS/mpich2/nemesis/bin/mpiexec
MY_PROG=$EBROOTFDTD_SOLUTIONS/bin/fdtd-engine-mpich2nem

INPUT="benchmark2.fsp"
NCORE=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))
$MPI -n $NCORE $MY_PROG ./${INPUT}

}}
```
## Templates

**Note:** This section is intended for use with the \"fdtd_solutions\" module and has not been adapted for \"lumerical\".

If you are performing a lot of simulations, you may find it inefficient to edit the job submission script for each simulation. You can use template submission scripts to improve this.

For example:

- Create directory `$HOME/bin` and put the main script `fdtd-run.sh` (see below) there.
- Create directory `$HOME/bin/templates` and put the job submission template script `fdtd-mpi-template.sh` and process template script `fdtd-process-template.sh` there.

`fdtd-mpi-template.sh` is basically a shell of the `fdtd_solutions.sh` script shown above and `fdtd-process-template.sh` determines the computing resources you need.

To submit a job, run

This will use the 32 cores on a single standard node. If you want to use more cores, request multiple nodes like so:

```{=mediawiki}
{{File  
  |name=fdtd-run.sh
  |lang="sh"
  |contents=

#!/bin/bash
# This script will create a Slurm-style job submission script for
# FDTD Solutions project files using the template provided in
# templates/fdtd-mpi-template.sh. Certain tags in the template
# file are replaced with values extracted from the project file.
#
# The calling convention for this script is:
#
# fdtd-run.sh [-nn <nodes>] fsp1 [fsp2 ... fspN]
#
# The arguments are as follows:
#
# -nn       The number of nodes to use for the job(s).
#           If no argument is given one node is used.
#
# fsp*      An FDTD Solutions project file. One is required, but
#           multiple can be specified on one command line.
#
##########################################################################

# Locate the directory of this script so we can find
# utility scripts and templates relative to this path.
module load fdtd_solutions
SCRIPTDIR=$EBROOTFDTD_SOLUTIONS/bin

# The location of the template file to use when submitting jobs.
# The line below can be changed to use your own template file.
TEMPLATE=../bin/templates/fdtd-mpi-template.sh

# Number of processes per node.
PROCS=32

# Number of nodes to use. Default is 1 if no -nn argument is given.
NODES=1
if [ "$1" = "-nn" ]; then
    NODES=$2
    shift
    shift
fi

# For each fsp file listed on the command line, generate the
# submission script and submit it with sbatch.
while(( $# > 0 ))
do

    # Generate the submission script by replacing the tokens in the template.
    # Additional arguments can be added to fdtd-process-template to fine-tune
    # the memory and time estimates. See comments in that file for details.
    SHELLFILE=${1%.fsp}.sh
    ../bin/templates/fdtd-process-template.sh -ms 500 $1 $TEMPLATE $((PROCS)) > $SHELLFILE
    TOTAL_MEM=$(head -n 1 $SHELLFILE)
    sed -i -e '1,1d' $SHELLFILE

    # Submit the job script.
    echo Submitting: $SHELLFILE
    echo Total Memory Required = $TOTAL_MEM
    sbatch --nodes=${NODES} --ntasks=${NODES} --cpus-per-task=${PROCS} --mem=${TOTAL_MEM} $SHELLFILE

    shift
done
}}
```
```{=mediawiki}
{{File  
  |name=fdtd-mpi-template.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --time=<hours>:<minutes>:<seconds>
#SBATCH --ntasks-per-node=1

module load fdtd_solutions
MPI=$EBROOTFDTD_SOLUTIONS/mpich2/nemesis/bin/mpiexec
MY_PROG=$EBROOTFDTD_SOLUTIONS/bin/fdtd-engine-mpich2nem

INPUT="<filename>"
NCORE=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))
$MPI -n $NCORE $MY_PROG ./${INPUT}
}}
```
`` sed 's/^.*=//'` ``

1.  Total memory required

TOTALMEM=\$(( ESTMEM \* MEMORY_SAFETY / 100 ))

1.  Memory required per process

PROCMEM=\$((TOTALMEM / PROCS)) if \[ \"\$PROCMEM\" -lt \"\$MEMORY_MIN\" \]; then

`   PROCMEM=$MEMORY_MIN`

fi

1.  Gridpoints

GRIDPTS=\`grep gridpoints \$1.tmp sed \'s/\^.\*=//\'\`

1.  Timesteps

TIMESTEPS=\`grep time_steps \$1.tmp sed \'s/\^.\*=//\'\`

1.  Estimated time

TIME=\$(( GRIDPTS \* TIMESTEPS / PROCS / RATE / 10000000 )) if \[ \"\$TIME\" -lt \"\$TIME_MIN\" \]; then

`   TIME=$TIME_MIN`

fi

HOUR=\$((TIME / 3600)) MINSEC=\$((TIME - HOUR \* 3600)) MIN=\$((MINSEC / 60)) SEC=\$((MINSEC - MIN \* 60))

echo \$TOTALMEM

1.  The replacements

sed -e \"s#`<total_memory>`{=html}#\$TOTALMEM#g\" \\

`   -e "s#``<processor_memory>`{=html}`#$PROCMEM#g" \`\
`   -e "s#``<hours>`{=html}`#$HOUR#g" \`\
`   -e "s#``<minutes>`{=html}`#$MIN#g" \`\
`   -e "s#``<seconds>`{=html}`#$SEC#g" \`\
`   -e "s#``<n>`{=html}`#$PROCS#g" \`\
`   -e "s#``<dir_fsp>`{=html}`#$DIRFSP#g" \`\
`   -e "s#``<filename>`{=html}`#$FILENAME#g" \`\
`   $2`

}}
