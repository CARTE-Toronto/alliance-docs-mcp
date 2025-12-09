---
title: "Dalton/en"
url: "https://docs.alliancecan.ca/wiki/Dalton/en"
category: "General"
last_modified: "2018-09-17T17:22:19Z"
page_id: 8908
display_title: "Dalton"
---

`<languages />`{=html}

# Introduction

The kernel of the Dalton2016 suite is the two powerful molecular electronic structure programs, Dalton and LSDalton. Together, the two programs provide extensive functionality for the calculations of molecular properties at the HF, DFT, MCSCF, and CC levels of theory. Many of these properties are only available in the Dalton2016 suite.

- Project web site: <http://daltonprogram.org/>
- Documentation: <http://daltonprogram.org/documentation/>
- Forum: <http://forum.daltonprogram.org/>

# Modules

``` bash
$ module load nixpkgs/16.09 intel/2016.4 openmpi/2.0.2 dalton/2017-alpha
```

Notice that `dalton/2017-alpha` depends on a non-default version of Open MPI. For more on the `module` command see [Using modules](https://docs.alliancecan.ca/Utiliser_des_modules/fr "Using modules"){.wikilink}.

# Usage

Here is an example:

- Dalton input file: `dft_rspexci_nosym.dal` (see the examples below).
- Molecule specification: `H2O_cc-pVDZ_nosym.mol` (see the examples below).
- To use the atomic basis set installed with the program, supply option `-b ${BASLIB}` to the command line (see the examples below).
- The number of processes can be set using a command line option or an environment variable:
  - Add the option `-N ${SLURM_NTASKS}` to the launcher command line (refer to **Script 1** in the examples below);
  - or, `export DALTON_NUM_MPI_PROCS=${SLURM_NTASKS}` (refer to **Script 2** in the examples below).

To run Dalton, load the module and use the launcher `dalton`:

``` bash
dalton -b ${BASLIB} -N ${SLURM_NTASKS}  -dal dft_rspexci_nosym.dal  -mol H2O_cc-pVDZ_nosym.mol 
```

or

``` bash
export DALTON_NUM_MPI_PROCS=${SLURM_NTASKS}
dalton -b ${BASLIB}  -dal dft_rspexci_nosym.dal  -mol H2O_cc-pVDZ_nosym.mol 
```

# Examples: scripts and input files {#examples_scripts_and_input_files}

## Example 1: dft_rspexci_nosym {#example_1_dft_rspexci_nosym}

`<tabs>`{=html}

`<tab name="INPUT">`{=html}

`</tab>`{=html}

`<tab name="MOLECULE">`{=html}

`</tab>`{=html}

`<tab name="Script 1">`{=html} ``{{File
  |name=run_dalton_job.sh
  |lang="bash"
  |contents=
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=3500M
#SBATCH --time=00-30:00

# Load the module: 

module load nixpkgs/16.09  intel/2016.4  openmpi/2.0.2 dalton/2017-alpha

# Setting the variables:

dltonlaun=dalton
dltonexec=dalton.x
daltoninput=dft_rspexci_nosym.dal
daltonmol=H2O_cc-pVDZ_nosym.mol

echo "Starting run at: `date`"

echo "Running the example: INPUT=${daltoninput} - Molecule=${daltonmol}"

${dltonlaun} -b ${BASLIB} -N ${SLURM_NTASKS}  -dal ${daltoninput}  -mol ${daltonmol}

echo "Program finished with exit code $? at: `date`"
}}``{=mediawiki} `</tab>`{=html}

`<tab name="Script 2">`{=html} ``{{File
  |name=run_dalton_job.sh
  |lang="bash"
  |contents=
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=3500M
#SBATCH --time=00-30:00

# Load the module: 

module load nixpkgs/16.09  intel/2016.4  openmpi/2.0.2 dalton/2017-alpha

# Setting the variables:

dltonlaun=dalton
dltonexec=dalton.x
daltoninput=dft_rspexci_nosym.dal
daltonmol=H2O_cc-pVDZ_nosym.mol

# Set the number of cores DALTON_NUM_MPI_PROCS to ${SLURM_NTASKS}

export DALTON_NUM_MPI_PROCS=${SLURM_NTASKS}

echo "Starting run at: `date`"

echo "Running the example: INPUT=${daltoninput} - Molecule=${daltonmol}"

${dltonlaun} -b ${BASLIB}  -dal ${daltoninput}  -mol ${daltonmol}

echo "Program finished with exit code $? at: `date`"
}}``{=mediawiki} `</tab>`{=html}

`</tabs>`{=html}

## Example 2: dft_rspexci_sym.dal {#example_2_dft_rspexci_sym.dal}

`<tabs>`{=html}

`<tab name="INPUT">`{=html}

`</tab>`{=html}

`<tab name="MOLECULE">`{=html}

`</tab>`{=html}

`<tab name="Script 1">`{=html} ``{{File
  |name=run_dalton_job.sh
  |lang="bash"
  |contents=
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=3500M
#SBATCH --time=00-30:00

# Load the module: 

module load nixpkgs/16.09  intel/2016.4  openmpi/2.0.2 dalton/2017-alpha

# Setting the variables:

dltonlaun=dalton
dltonexec=dalton.x
daltoninput=dft_rspexci_sym.dal
daltonmol=H2O_cc-pVDZ_sym.mol

echo "Starting run at: `date`"

echo "Running the example: INPUT=${daltoninput} - Molecule=${daltonmol}"

${dltonlaun} -b ${BASLIB} -N ${SLURM_NTASKS}  -dal ${daltoninput}  -mol ${daltonmol}

echo "Program finished with exit code $? at: `date`"
}}``{=mediawiki} `</tab>`{=html}

`<tab name="Script 2">`{=html} ``{{File
  |name=run_dalton_job.sh
  |lang="bash"
  |contents=
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=3500M
#SBATCH --time=00-30:00

# Load the module:

module load nixpkgs/16.09  intel/2016.4  openmpi/2.0.2 dalton/2017-alpha

# Setting the variables:

dltonlaun=dalton
dltonexec=dalton.x
daltoninput=dft_rspexci_sym.dal
daltonmol=H2O_cc-pVDZ_sym.mol

# Set the number of cores DALTON_NUM_MPI_PROCS to ${SLURM_NTASKS}

export DALTON_NUM_MPI_PROCS=${SLURM_NTASKS}

echo "Starting run at: `date`"

echo "Running the example: INPUT=${daltoninput} - Molecule=${daltonmol}"

${dltonlaun} -b ${BASLIB}  -dal ${daltoninput}  -mol ${daltonmol}

echo "Program finished with exit code $? at: `date`"
}}``{=mediawiki} `</tab>`{=html}

`</tabs>`{=html}
