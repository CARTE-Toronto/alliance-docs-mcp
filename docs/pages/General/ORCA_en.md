---
title: "ORCA/en"
url: "https://docs.alliancecan.ca/wiki/ORCA/en"
category: "General"
last_modified: "2025-03-11T18:18:52Z"
page_id: 4592
display_title: "ORCA"
---

`<languages />`{=html}

## Introduction

ORCA is a flexible, efficient and easy-to-use general-purpose tool for quantum chemistry with specific emphasis on spectroscopic properties of open-shell molecules. It features a wide variety of standard quantum chemical methods ranging from semiempirical methods to DFT to single- and multireference correlated `<i>`{=html}ab initio`</i>`{=html} methods. It can also treat environmental and relativistic effects.

## Licensing

If you wish to use prebuilt ORCA executables:

1.  You have to register at <https://orcaforum.kofo.mpg.de/>.
2.  You will receive a first email to verify the email address and activate the account. Follow the instructions in that email.
3.  Once the registration is complete, you will get a `<b>`{=html}second email`</b>`{=html} stating that the \"`<i>`{=html}registration for ORCA download and usage has been completed`</i>`{=html}\".
4.  [ Contact us](https://docs.alliancecan.ca/Technical_support " Contact us"){.wikilink} requesting access to ORCA with a copy of the `<b>`{=html}second email`</b>`{=html}.

## ORCA versions {#orca_versions}

### ORCA 6 {#orca_6}

A module **orca/6.0.1** is available under the environment **StdEnv/2023**. To load this module, use:

`module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 orca/6.0.1`

There is another module **orca/6.0.0**. However, ORCA users should use the latest version **orca/6.0.1** as it addresses some bugs of the first release **6.0.0**.

**Note:** This version of ORCA includes xtb 6.7.1.

### ORCA 5 {#orca_5}

Versions 5.0.1 through 5.0.3 have some bugs that were fixed in version 5.0.4, most notably a [bug involving D4 dispersion gradients](https://orcaforum.kofo.mpg.de/viewtopic.php?f=56&t=9985). We therefore recommend that you use the version 5.0.4 instead of any earlier 5.0.x version. Versions 5.0.1, 5.0.2 and 5.0.3 are in our software stack but might be removed in the future.

To load version 5.0.4, use

`module load StdEnv/2020 gcc/10.3.0 openmpi/4.1.1 orca/5.0.4`

### ORCA 4 {#orca_4}

To load version 4.2.1, use

`module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 orca/4.2.1`

or

`module load nixpkgs/16.09 gcc/7.3.0 openmpi/3.1.4 orca/4.2.1`

## Setting ORCA input files {#setting_orca_input_files}

In addition to the different keywords required to run a given simulation, you should make sure to set two additional parameters:

- number of CPUs

<!-- -->

- maxcore

## Using the software {#using_the_software}

To see which versions of ORCA are currently available, type `module spider orca`. For detailed information about a specific version, including the other modules that must be loaded first, use the module\'s full name. For example, `module spider orca/4.0.1.2`.

See [Using modules](https://docs.alliancecan.ca/Using_modules "Using modules"){.wikilink} for general guidance.

### Job submission {#job_submission}

For a general discussion about submitting jobs, see [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink}.

`<b>`{=html}NOTE`</b>`{=html}: If you run into MPI errors with some of the ORCA executables, you can try to define the following variables:

`export OMPI_MCA_mtl='^mxm'`\
`export OMPI_MCA_pml='^yalla'`

The following is a job script to run ORCA using [MPI](https://docs.alliancecan.ca/MPI "MPI"){.wikilink}. Note that, unlike most MPI programs, ORCA is not started with a parallel launch command such as `mpirun` or `srun`, but requires the full path to the program, which is given by `$EBROOTORCA`.

Example of the input file, benzene.inp:

### Notes

- To make sure that the program runs efficiently and makes use of all the resources or the cores asked for in your job script, please add this line `%pal nprocs ``<ncores>`{=html}` end` to your input file as shown in the above example. Replace `<ncores>`{=html} by the number of cores you used in your script.

<!-- -->

- If you want to restart a calculation, delete the file `*.hostnames` (e.g `benzene.hostnames` in the above example) before you submit the followup job. If you do not, the job is likely to fail with the error message `All nodes which are allocated for this job are already filled.`

### (Sep. 6 2019) Temporary fix to OpenMPI version inconsistency issue {#sep._6_2019_temporary_fix_to_openmpi_version_inconsistency_issue}

For some type of calculations (DLPNO-STEOM-CCSD in particular), you could receive unknown openmpi related fatal errors. This could be due to using an older version of openmpi (`<i>`{=html}i.e.`</i>`{=html} 3.1.2 as suggested by \'module\' for both orca/4.1.0 and 4.2.0) than recommended officially (3.1.3 for orca/4.1.0 and 3.1.4 for orca/4.2.0). To temporarily fix this issue, one can build a custom version of openmpi.

The following two commands prepare a custom openmpi/3.1.4 for orca/4.2.0:

`       module load gcc/7.3.0`\
`       eb OpenMPI-3.1.2-GCC-7.3.0.eb --try-software-version=3.1.4`

When the building is finished, you can load the custom openmpi using module

`       module load openmpi/3.1.4`

At this step, one can manually install orca/4.2.0 binaries from the official forum under the home directory after finishing the registration on the official orca forum and being granted access to the orca program on our clusters.

Additional notes from the contributor:

This is a `<b>`{=html}temporary`</b>`{=html} fix prior to the official upgrade of openmpi on our clusters. Please remember to delete the manually installed orca binaries once the official openmpi version is up to date.

The compiling command does not seem to apply to openmpi/2.1.x.

## Using NBO with ORCA {#using_nbo_with_orca}

To run NBO with ORCA, one needs to have access to NBO. On our clusters, NBO is not available as a separate module. However, it is possible to access it via the Gaussian modules which are installed on [Cedar](https://docs.alliancecan.ca/Cedar "Cedar"){.wikilink} and [Graham](https://docs.alliancecan.ca/Graham "Graham"){.wikilink}. Users interested to use NBO with ORCA should have access to ORCA and Gaussian. To get access to Gaussian, you can follow the steps discussed in this [page](https://docs.alliancecan.ca/Gaussian#License_agreement "page"){.wikilink}.

### Script example {#script_example}

The name of the input file (in this next example `<i>`{=html}orca_input.inp`</i>`{=html} should contain the keyword `<b>`{=html}NBO`</b>`{=html}.

```{=mediawiki}
{{File
  |name=run_orca-nbo.sh
  |lang="bash"
  |contents=
#!/bin/bash
#SBATCH --account=def-youPIs
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4000
#SBATCH --time=0-3:00:00

# Load the modules:

module load StdEnv/2020  gcc/10.3.0  openmpi/4.1.1 orca/5.0.4
module load gaussian/g16.c01

export GENEXE=`which gennbo.i4.exe`
export NBOEXE=`which nbo7.i4.exe`

${EBROOTORCA}/orca orca_input.inp > orca_output.out

}}
```
## Related links {#related_links}

- [ORCA tutorials](https://www.orcasoftware.de/tutorials_orca/)
- [ORCA Forum](https://orcaforum.kofo.mpg.de/app.php/portal)
