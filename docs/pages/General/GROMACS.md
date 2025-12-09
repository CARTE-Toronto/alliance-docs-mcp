---
title: "GROMACS/en"
url: "https://docs.alliancecan.ca/wiki/GROMACS/en"
category: "General"
last_modified: "2025-08-29T12:31:02Z"
page_id: 4167
display_title: "GROMACS"
---

`<languages />`{=html}

# General

[GROMACS](http://www.gromacs.org/) is a versatile package to perform molecular dynamics for systems with hundreds to millions of particles. It is primarily designed for biochemical molecules like proteins, lipids and nucleic acids that have a lot of complicated bonded interactions, but since GROMACS is extremely fast at calculating the nonbonded interactions (that usually dominate simulations) many groups are also using it for research on non-biological systems, e.g. polymers.

## Strengths

- GROMACS provides extremely high performance compared to all other programs.
- Since GROMACS 4.6, we have excellent CUDA-based GPU acceleration on GPUs that have Nvidia compute capability \>= 2.0 (e.g. Fermi or later).
- GROMACS comes with a large selection of flexible tools for trajectory analysis.
- GROMACS can be run in parallel, using either the standard MPI communication protocol, or via our own \"Thread MPI\" library for single-node workstations.
- GROMACS is free software, available under the GNU Lesser General Public License (LGPL), version 2.1.

## Weak points {#weak_points}

- To get very high simulation speed, GROMACS does not do much additional analysis and / or data collection on the fly. It may be a challenge to obtain somewhat non-standard information about the simulated system from a GROMACS simulation.

<!-- -->

- Different versions may have significant differences in simulation methods and default parameters. Reproducing results of older versions with a newer version may not be straightforward.

<!-- -->

- Additional tools and utilities that come with GROMACS are not always of the highest quality, may contain bugs and may implement poorly documented methods. Reconfirming the results of such tools with independent methods is always a good idea.

## GPU support {#gpu_support}

The top part of any log file will describe the configuration, and in particular whether your version has GPU support compiled in. GROMACS will automatically use any GPUs it finds.

GROMACS uses both CPUs and GPUs; it relies on a reasonable balance between CPU and GPU performance.

The new neighbour structure required the introduction of a new variable called \"cutoff-scheme\" in the mdp file. The behaviour of older GROMACS versions (before 4.6) corresponds to `cutoff-scheme = group`, while in order to use GPU acceleration you must change it to `cutoff-scheme = verlet`, which has become the new default in version 5.0.

# Quickstart guide {#quickstart_guide}

This section summarizes configuration details.

## Environment modules {#environment_modules}

The following versions have been installed:

`<tabs>`{=html} `<tab name="StdEnv/2023">`{=html}

  GROMACS version   modules for running on CPUs                           modules for running on GPUs (CUDA)                              Notes
  ----------------- ----------------------------------------------------- --------------------------------------------------------------- -----------------------
  gromacs/2024.4    `StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4`   `StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs/2024.4`   GCC, FlexiBLAS & FFTW
  gromacs/2024.1    `StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.1`   `StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs/2024.1`   GCC, FlexiBLAS & FFTW
  gromacs/2023.5    `StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2023.5`   `StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs/2023.5`   GCC, FlexiBLAS & FFTW
  gromacs/2023.3    `StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2023.3`   `StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs/2023.3`   GCC, FlexiBLAS & FFTW
                                                                                                                                          

`</tab>`{=html} `<tab name="StdEnv/2020">`{=html}

  GROMACS version   modules for running on CPUs                            modules for running on GPUs (CUDA)                                                                           Notes
  ----------------- ------------------------------------------------------ ------------------------------------------------------------------------------------------------------------ -----------------------
  gromacs/2023.2    `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2023.2`   `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2023.2`                                               GCC, FlexiBLAS & FFTW
  gromacs/2023      `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2023`     `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2023`                                                 GCC, FlexiBLAS & FFTW
  gromacs/2022.3    `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2022.3`   `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2022.3`                                               GCC, FlexiBLAS & FFTW
  gromacs/2022.2    `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2022.2`                                                                                                                GCC, FlexiBLAS & FFTW
  gromacs/2021.6    `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2021.6`   `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2021.6`                                               GCC, FlexiBLAS & FFTW
  gromacs/2021.4    `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2021.4`   `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2021.4` ![warning](https://docs.alliancecan.ca/Warning_sign_16px.png "warning")   GCC, FlexiBLAS & FFTW
  gromacs/2021.2                                                           `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2021.2` ![warning](https://docs.alliancecan.ca/Warning_sign_16px.png "warning")   GCC, FlexiBLAS & FFTW
  gromacs/2021.2    `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2021.2`   `StdEnv/2020 gcc/9.3.0 cuda/11.0 openmpi/4.0.3 gromacs/2021.2` ![warning](https://docs.alliancecan.ca/Warning_sign_16px.png "warning")   GCC & MKL
  gromacs/2020.6    `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2020.6`   `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2020.6` ![warning](https://docs.alliancecan.ca/Warning_sign_16px.png "warning")   GCC, FlexiBLAS & FFTW
  gromacs/2020.4                                                           `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs/2020.4` ![warning](https://docs.alliancecan.ca/Warning_sign_16px.png "warning")   GCC, FlexiBLAS & FFTW
  gromacs/2020.4    `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs/2020.4`   `StdEnv/2020 gcc/9.3.0 cuda/11.0 openmpi/4.0.3 gromacs/2020.4` ![warning](https://docs.alliancecan.ca/Warning_sign_16px.png "warning")   GCC & MKL

`</tab>`{=html} `<tab name="StdEnv/2018.3">`{=html}

  GROMACS version   modules for running on CPUs                              modules for running on GPUs (CUDA)                                                                                 Notes
  ----------------- -------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------ --------------
  gromacs/2020.2    `StdEnv/2018.3 gcc/7.3.0 openmpi/3.1.2 gromacs/2020.2`   `StdEnv/2018.3 gcc/7.3.0 cuda/10.0.130 openmpi/3.1.2 gromacs/2020.2` ![warning](https://docs.alliancecan.ca/Warning_sign_16px.png "warning")   GCC & MKL
  gromacs/2019.6    `StdEnv/2018.3 gcc/7.3.0 openmpi/3.1.2 gromacs/2019.6`   `StdEnv/2018.3 gcc/7.3.0 cuda/10.0.130 openmpi/3.1.2 gromacs/2019.6`                                               GCC & MKL
  gromacs/2019.3    `StdEnv/2018.3 gcc/7.3.0 openmpi/3.1.2 gromacs/2019.3`   `StdEnv/2018.3 gcc/7.3.0 cuda/10.0.130 openmpi/3.1.2 gromacs/2019.3`                                               GCC & MKL  ‡
  gromacs/2018.7    `StdEnv/2018.3 gcc/7.3.0 openmpi/3.1.2 gromacs/2018.7`   `StdEnv/2018.3 gcc/7.3.0 cuda/10.0.130 openmpi/3.1.2 gromacs/2018.7`                                               GCC & MKL

`</tab>`{=html} `<tab name="StdEnv/2016.4">`{=html}

  GROMACS version   modules for running on CPUs                              modules for running on GPUs (CUDA)                                    Notes
  ----------------- -------------------------------------------------------- --------------------------------------------------------------------- -----------------------
  gromacs/2018.3    `StdEnv/2016.4 gcc/6.4.0 openmpi/2.1.1 gromacs/2018.3`   `StdEnv/2016.4 gcc/6.4.0 cuda/9.0.176 openmpi/2.1.1 gromacs/2018.3`   GCC & FFTW
  gromacs/2018.2    `StdEnv/2016.4 gcc/6.4.0 openmpi/2.1.1 gromacs/2018.2`   `StdEnv/2016.4 gcc/6.4.0 cuda/9.0.176 openmpi/2.1.1 gromacs/2018.2`   GCC & FFTW
  gromacs/2018.1    `StdEnv/2016.4 gcc/6.4.0 openmpi/2.1.1 gromacs/2018.1`   `StdEnv/2016.4 gcc/6.4.0 cuda/9.0.176 openmpi/2.1.1 gromacs/2018.1`   GCC & FFTW
  gromacs/2018      `StdEnv/2016.4 gromacs/2018`                             `StdEnv/2016.4 cuda/9.0.176 gromacs/2018`                             Intel & MKL
  gromacs/2016.5    `StdEnv/2016.4 gcc/6.4.0 openmpi/2.1.1 gromacs/2016.5`   `StdEnv/2016.4 gcc/6.4.0 cuda/9.0.176 openmpi/2.1.1 gromacs/2016.5`   GCC & FFTW
  gromacs/2016.3    `StdEnv/2016.4 gromacs/2016.3`                           `StdEnv/2016.4 cuda/8.0.44 gromacs/2016.3`                            Intel & MKL
  gromacs/5.1.5     `StdEnv/2016.4 gromacs/5.1.5`                            `StdEnv/2016.4 cuda/8.0.44 gromacs/5.1.5`                             Intel & MKL
  gromacs/5.1.4     `StdEnv/2016.4 gromacs/5.1.4`                            `StdEnv/2016.4 cuda/8.0.44 gromacs/5.1.4`                             Intel & MKL
  gromacs/5.0.7     `StdEnv/2016.4 gromacs/5.0.7`                            `StdEnv/2016.4 cuda/8.0.44 gromacs/5.0.7`                             Intel & MKL
  gromacs/4.6.7     `StdEnv/2016.4 gromacs/4.6.7`                            `StdEnv/2016.4 cuda/8.0.44 gromacs/4.6.7`                             Intel & MKL
  gromacs/4.6.7     `StdEnv/2016.4 gcc/5.4.0 openmpi/2.1.1 gromacs/4.6.7`    `StdEnv/2016.4 gcc/5.4.0 cuda/8.0 openmpi/2.1.1 gromacs/4.6.7`        GCC & MKL & ThreadMPI

`</tab>`{=html} `</tabs>`{=html}

**Notes:**

- ![warning](https://docs.alliancecan.ca/Warning_sign_16px.png "warning") GROMACS versions 2020.0 up to and including 2021.5 contain a bug when used on GPUs of Volta or newer generations (i.e. V100, T4, A100, and H100) with `mdrun` option `-update gpu` that could have perturbed the virial calculation and, in turn, led to incorrect pressure coupling. The GROMACS developers state in the 2021.6 Release Notes:[^1]
  > *The GPU update is not enabled by default, so the error can only appear in simulations where it \[the `-update gpu` option\] was manually selected, and even in this case the error might be rare since we have not observed it in practice in the testing we have performed.*

  Further discussion of this bug can be found in the GitLab issue #4393 of the GROMACS project.[^2]
- Version 2020.4 and newer have been compiled for the new [Standard software environment](https://docs.alliancecan.ca/Standard_software_environments "Standard software environment"){.wikilink} `StdEnv/2020`.
- Version 2018.7 and newer have been compiled with GCC compilers and the MKL-library, as they run a bit faster.
- Older versions have been compiled with either with GCC compilers and FFTW or Intel compilers, using Intel MKL and Open MPI 2.1.1 libraries from the default environment as indicated in the table above.
- CPU (non-GPU) versions are available in both single- and double precision, with the exception of 2019.3 (**‡**), where double precision is not available for AVX512.

These modules can be loaded by using a `module load` command with the modules as stated in the second column in the above table. For example:

`$ module load  StdEnv/2023  gcc/12.3   openmpi/4.1.5  gromacs/2024.4`\
`or `\
`$ module load  StdEnv/2020  gcc/9.3.0  openmpi/4.0.3  gromacs/2023.2`

These versions are also available with GPU support, albeit only with single precision. In order to load the GPU enabled version, the `cuda` module needs to be loaded first. The modules needed are listed in the third column of above table, e.g.:

`$ module load  StdEnv/2023  gcc/12.3  openmpi/4.1.5  cuda/12.2  gromacs/2024.4`\
`or`\
`$ module load  StdEnv/2020  gcc/9.3.0  cuda/11.4  openmpi/4.0.3  gromacs/2023.2 `

For more information on environment modules, please refer to the [Using modules](https://docs.alliancecan.ca/Using_modules "Using modules"){.wikilink} page.

## Suffixes

### GROMACS 5.x, 2016.x and newer {#gromacs_5.x_2016.x_and_newer}

GROMACS 5 and newer releases consist of only four binaries that contain the full functionality. All GROMACS tools from previous versions have been implemented as sub-commands of the gmx binaries. Please refer to [GROMACS 5.0 Tool Changes](http://www.gromacs.org/Documentation/How-tos/Tool_Changes_for_5.0) and the [GROMACS documentation manuals](http://manual.gromacs.org/documentation/) for your version.

:\* **`gmx`** - mixed (\"single\") precision GROMACS with OpenMP (threading) but without MPI.

:\* **`gmx_mpi`** - mixed (\"single\") precision GROMACS with OpenMP and MPI.

:\* **`gmx_d`** - double precision GROMACS with OpenMP but without MPI.

:\* **`gmx_mpi_d`** - double precision GROMACS with OpenMP and MPI.

### GROMACS 4.6.7 {#gromacs_4.6.7}

- The double precision binaries have the suffix `_d`.
- The parallel single and double precision `mdrun` binaries are:

:\* **`mdrun_mpi`**

:\* **`mdrun_mpi_d`**

## Submission scripts {#submission_scripts}

Please refer to the page [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink} for help on using the SLURM workload manager.

### Serial jobs {#serial_jobs}

Here\'s a simple job script for serial mdrun:

```{=mediawiki}
{{File
  |name=serial_gromacs_job.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --time=0-0:30         # time limit (D-HH:MM)
#SBATCH --mem-per-cpu=1000M   # memory per CPU (in MB)
module purge  
module load  StdEnv/2023  gcc/12.3  openmpi/4.1.5  gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
gmx mdrun -nt 1 -deffnm em
}}
```
This will run the simulation of the molecular system in the file `em.tpr`.

### Whole nodes {#whole_nodes}

Commonly, the systems simulated with GROMACS are so large that you should use a number of whole nodes for the simulation.

Generally, the product of `--ntasks-per-node` and `--cpus-per-task` should match the number of CPU cores of the cluster's compute nodes. See section [Performance and benchmarking](https://docs.alliancecan.ca/GROMACS#Performance_and_benchmarking "Performance and benchmarking"){.wikilink} below.

On clusters with a large number of CPU cores (e.g. 192) per compute node, domain decomposition can become a limiting factor when choosing `--ntasks-per-node`. The larger a system is, the more it can be divided into smaller regions, allowing larger `--ntasks-per-node` values. If GROMACS reports an error about domain decomposition being impossible given the system size and requested number of domains, halve `--ntasks-per-node` and double `--cpus-per-task`.

`<tabs>`{=html} `<tab name="Narval">`{=html} `{{File
  |name=gromacs_whole_node_narval.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32     # request 32 MPI tasks per node
#SBATCH --cpus-per-task=2        # 2 OpenMP threads per MPI task => total: 32 x 2 = 64 CPUs/node
#SBATCH --mem-per-cpu=2000M      # memory per CPU (in MB)
#SBATCH --time=0-01:00           # time limit (D-HH:MM)
module purge  
module load  StdEnv/2023  gcc/12.3  openmpi/4.1.5  gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
 
srun --cpus-per-task=$OMP_NUM_THREADS gmx_mpi mdrun -deffnm md
}}`{=mediawiki}`</tab>`{=html} `<tab name="Rorqual">`{=html} `{{File
  |name=gromacs_whole_node_rorqual.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=96     # request 96 MPI tasks per node
#SBATCH --cpus-per-task=2        # 2 OpenMP threads per MPI task => total: 96 x 2 = 192 CPUs/node
#SBATCH --mem-per-cpu=2000M      # memory per CPU (in MB)
#SBATCH --time=0-01:00           # time limit (D-HH:MM)
module purge  
module load  StdEnv/2023  gcc/12.3  openmpi/4.1.5  gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun --cpus-per-task=$OMP_NUM_THREADS gmx_mpi mdrun -deffnm md
}}`{=mediawiki}`</tab>`{=html} `<tab name="Fir">`{=html} `{{File
  |name=gromacs_whole_node_fir.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=96     # request 96 MPI tasks per node
#SBATCH --cpus-per-task=2        # 2 OpenMP threads per MPI task => total: 96 x 2 = 192 CPUs/node
#SBATCH --mem-per-cpu=2000M      # memory per CPU (in MB)
#SBATCH --time=0-01:00           # time limit (D-HH:MM)
module purge  
module load  StdEnv/2023  gcc/12.3  openmpi/4.1.5  gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun --cpus-per-task=$OMP_NUM_THREADS gmx_mpi mdrun -deffnm md
}}`{=mediawiki}`</tab>`{=html} `<tab name="Nibi">`{=html} `{{File
  |name=gromacs_whole_node_nibi.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=96     # request 96 MPI tasks per node
#SBATCH --cpus-per-task=2        # 2 OpenMP threads per MPI task => total: 96 x 2 = 192 CPUs/node
#SBATCH --mem-per-cpu=2000M      # memory per CPU (in MB)
#SBATCH --time=0-01:00           # time limit (D-HH:MM)
module purge  
module load  StdEnv/2023  gcc/12.3  openmpi/4.1.5  gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun --cpus-per-task=$OMP_NUM_THREADS gmx_mpi mdrun -deffnm md
}}`{=mediawiki}`</tab>`{=html} `<tab name="Trillium">`{=html} `{{File
  |name=gromacs_whole_node_trillium.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=96     # request 96 MPI tasks per node
#SBATCH --cpus-per-task=2        # 2 OpenMP threads per MPI task => total: 96 x 2 = 192 CPUs/node
#SBATCH --mem-per-cpu=2000M      # memory per CPU (in MB)
#SBATCH --time=0-01:00           # time limit (D-HH:MM)
module purge  
module load  StdEnv/2023  gcc/12.3  openmpi/4.1.5  gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun --cpus-per-task=$OMP_NUM_THREADS gmx_mpi mdrun -deffnm md
}}`{=mediawiki}`</tab>`{=html} `</tabs>`{=html}

### GPU job {#gpu_job}

Please read [Using GPUs with Slurm](https://docs.alliancecan.ca/Using_GPUs_with_Slurm "Using GPUs with Slurm"){.wikilink} for general information on using GPUs on our systems.

This is a job script for mdrun using 4 OpenMP threads and one GPU: `{{File
  |name=gpu_gromacs_job.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --gpus-per-node=1        # request 1 GPU per node
#SBATCH --cpus-per-task=4        # number of OpenMP threads per MPI process
#SBATCH --mem-per-cpu=2000M      # memory limit per CPU core (megabytes)
#SBATCH --time=0:30:00           # time limit (D-HH:MM:ss)
module purge  
module load StdEnv/2023  gcc/12.3  openmpi/4.1.5  cuda/12.2  gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

gmx mdrun -ntomp ${SLURM_CPUS_PER_TASK:-1} -deffnm md
}}`{=mediawiki}

#### Notes on running GROMACS on GPUs {#notes_on_running_gromacs_on_gpus}

Note that using more than a single GPU usually leads to poor efficiency. Carefully test and compare multi-GPU and single-GPU performance before deciding to use more than one GPU for your simulations.

- ![warning](https://docs.alliancecan.ca/Warning_sign_16px.png "warning") GROMACS versions 2020.0 up to and including 2021.5 contain a bug when used on GPUs of Volta or newer generations (i.e. V100, T4 and A100) with `mdrun` option `-update gpu` that could have perturbed the virial calculation and, in turn, led to incorrect pressure coupling. The GROMACS developers state in the 2021.6 Release Notes:[^3]
  > *The GPU update is not enabled by default, so the error can only appear in simulations where it was manually selected, and even in this case the error might be rare since we have not observed it in practice in the testing we have performed.*

  Further discussion of this bug can be found in the GitLab issue #4393 of the GROMACS project.[^4]
- Our clusters have differently configured GPU nodes.\
  On the page [Using GPUs with Slurm#Available GPUs](https://docs.alliancecan.ca/Using_GPUs_with_Slurm#Available_GPUs "Using GPUs with Slurm#Available GPUs"){.wikilink} you can find more information about the different node configurations (GPU models and number of GPUs and CPUs per node).
- GROMACS imposes a number of constraints for choosing the number of GPUs, tasks (MPI ranks) and OpenMP threads.\
  For GROMACS 2018.2 the constraints are:

::\* The number of `--tasks-per-node` always needs to be the same as, or a multiple of the number of GPUs (`--gpus-per-node`).

::\* GROMACS will not run GPU runs with only 1 OpenMP thread unless forced by setting the `-ntomp` option.\
According to GROMACS developers, the optimum number of `--cpus-per-task` is between 2 and 6.

- Avoid using a larger fraction of CPUs and memory than the fraction of GPUs you have requested in a node.

You can explore some benchmark results on our [MDBench portal](https://mdbench.ace-net.ca/mdbench/bform/?software_contains=GROMACS.cuda&software_id=&module_contains=&module_version=&site_contains=&gpu_model=&cpu_model=&arch=&dataset=6n4o).

#### Running multiple simulations on a GPU {#running_multiple_simulations_on_a_gpu}

GROMACS and other MD simulation programs are unable to fully use recent GPU models such as the Nvidia A100 and H100 unless the molecular system is very large (millions of atoms). Running a typical simulation on such a GPU wastes a significant fraction of the allocated computational resources.

There are two recommended solutions to this problem. The first one is to run multiple simulations on a single GPU using `mdrun -multidir` as described below. This is the preferred solution if you run multiple similar simulations, for instance:

- Repeating the same simulation to acquire more conformational space sampling
- Simulating multiple protein variants, multiple small ligands in complex with the same protein, multiple temperatures or ionic concentrations, etc.
- Ensemble-based simulations such as replica exchange

Similar simulations are needed to ensure proper load balancing. If the simulations are dissimilar, some will progress faster and finish earlier than others, leading to idle resources.

The following job script runs three similar simulations in separate directories (`sim1`, `sim2`, `sim3`) using a single GPU. If you change the number of simulations, make sure to adjust `--ntasks-per-node` and `--cpus-per-task`: there should be one task per simulation, while the total number of CPU cores should remain constant.

`<tabs>`{=html} `<tab name="Narval">`{=html} `{{File
  |name=gpu_gromacs_job_multidir.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --gpus-per-node=1        # request 1 GPU per node
#SBATCH --ntasks-per-node=3      # number of MPI processes and simulations
#SBATCH --cpus-per-task=4        # number of OpenMP threads per MPI process
#SBATCH --mem-per-cpu=2000M      # memory limit per CPU core (megabytes)
#SBATCH --time=0:30:00           # time limit (D-HH:MM:ss)

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs/2024.4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun gmx_mpi mdrun -ntomp ${SLURM_CPUS_PER_TASK:-1} -deffnm md \
    -multidir sim1 sim2 sim3
}}`{=mediawiki}`</tab>`{=html} `</tabs>`{=html}

The second solution is to use a [ MIG](https://docs.alliancecan.ca/Multi-Instance_GPU " MIG"){.wikilink} instance (a fraction of a GPU) rather than a full GPU. This is the preferred solution if you have a single simulation or if your simulations are dissimilar, for instance:

- Systems with different sizes (more than a 10 % difference in the numbers of atoms)
- Systems with different shapes or compositions, such as a membrane-bound versus a soluble protein

![warning](https://docs.alliancecan.ca/Warning_sign_16px.png "warning") Note that [Hyper-Q / MPS](https://docs.alliancecan.ca/Hyper-Q_/_MPS "Hyper-Q / MPS"){.wikilink} should never be used with GROMACS. The built-in `-multidir` option achieves the same functionality more efficiently.

# Usage

<div style="color: red; border: 1px dashed #2f6fab">

More content for this section will be added at a later time.

</div>

## System preparation {#system_preparation}

In order to run a simulation, one needs to create a *tpr* file (portable binary run input file). This file contains the starting structure of the simulation, the molecular topology and all the simulation parameters.

*Tpr* files are created with the `gmx grompp` command (or simply `grompp` for versions older than 5.0). Therefore one needs the following files:

- The coordinate file with the starting structure. GROMACS can read the starting structure from various file formats, such as *.gro*, *.pdb* or *.cpt* (checkpoint).
- The (system) topology (*.top*)) file. It defines which force field is used and how the force field parameters are applied to the simulated system. Often the topologies for individual parts of the simulated system (e.g. molecules) are placed in separate *.itp* files and included in the *.top* file using a `#include` directive.
- The run parameter (*.mdp*) file. See the GROMACS user guide for a detailed description of the options.

*Tpr* files are portable, that is they can be *grompp*\'ed on one machine, copied over to a different machine and used as an input file for *mdrun*. One should always use the same version for both *grompp* and *mdrun*. Although *mdrun* is able to use *tpr* files that have been created with an older version of *grompp*, this can lead to unexpected simulation results.

## Running simulations {#running_simulations}

MD Simulations often take much longer than the maximum walltime for a job to complete and therefore need to be restarted. To minimize the time a job needs to wait before it starts, you should maximize [the number of nodes you have access to](https://docs.alliancecan.ca/Job_scheduling_policies#Percentage_of_the_nodes_you_have_access_to "the number of nodes you have access to"){.wikilink} by choosing a shorter running time for your job. Requesting a walltime of 24 hours or 72 hours (three days) is often a good trade-off between waiting time and running time.

You should use the `mdrun` parameter `-maxh` to tell the program the requested walltime so that it gracefully finishes the current timestep when reaching 99% of this walltime. This causes `mdrun` to create a new checkpoint file at this final timestep and gives it the chance to properly close all output files (trajectories, energy- and log-files, etc.).

For example use `#SBATCH --time=24:00` along with `gmx mdrun -maxh 24 ...` or `#SBATCH --time=3-00:00` along with `gmx mdrun -maxh 72 ...`.

```{=mediawiki}
{{File
  |name=gromacs_job.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --nodes=1                # number of Nodes
#SBATCH --tasks-per-node=32      # number of MPI processes per node
#SBATCH --mem-per-cpu=4000       # memory limit per CPU (megabytes)
#SBATCH --time=24:00:00          # time limit (D-HH:MM:ss)
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun  gmx_mpi  mdrun  -deffnm md  -maxh 24
}}
```
### Restarting simulations {#restarting_simulations}

You can restart a simulation by using the same `mdrun` command as the original simulation and adding the `-cpi state.cpt` parameter where `state.cpt` is the filename of the most recent checkpoint file. Mdrun will by default (since version 4.5) try to append to the existing files (trajectories, energy- and log-files, etc.). GROMACS will check the consistency of the output files and - if needed - discard timesteps that are newer than that of the checkpoint file.

Using the `-maxh` parameter ensures that the checkpoint and output files are written in a consistent state when the simulation reaches the time limit.

The GROMACS manual contains more detailed information [^5] [^6].

```{=mediawiki}
{{File
  |name=gromacs_job_restart.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --nodes=1                # number of Nodes
#SBATCH --tasks-per-node=32      # number of MPI processes per node
#SBATCH --mem-per-cpu=4000       # memory limit per CPU (megabytes)
#SBATCH --time=24:00:00          # time limit (D-HH:MM:ss)
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun  gmx_mpi  mdrun  -deffnm md  -maxh 24.0  -cpi md.cpt
}}
```
### Checkpointing simulations {#checkpointing_simulations}

You can use GROMACS' ability to restart a simulation to split a long simulation over multiple short jobs. Shorter jobs wait less in the queue. In particular, those that request 3 hours or less are eligible for backfill scheduling. (See our [ job scheduling policies](https://docs.alliancecan.ca/Job_scheduling_policies " job scheduling policies"){.wikilink}.) This is especially useful if your research group has only a default resource allocation (e.g. `def-sponsor`) on the cluster, but will benefit even those with competitive resource allocations (e.g. `rrg-sponsor`).

By using a [ job array](https://docs.alliancecan.ca/Job_arrays " job array"){.wikilink}, you can automate checkpointing. With an array job script such as the following, a single `sbatch` call submits multiple short jobs, but only the first one is eligible to start. As soon as this first job has completed, the next one becomes eligible to start and resume your simulation. This process repeats until all jobs are complete or the simulation is finished, at which point any remaining pending jobs are automatically cancelled.

`<tabs>`{=html} `<tab name="Whole nodes (Narval)">`{=html} `{{File
  |name=gromacs_job_checkpoint.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32     # request 32 MPI tasks per node
#SBATCH --cpus-per-task=2        # 2 OpenMP threads per MPI task
#SBATCH --mem-per-cpu=2000M      # memory per CPU (in MB)
#SBATCH --time=03:00:00          # time limit (D-HH:MM:ss)
#SBATCH --array=1-20%1           # job range, running only 1 at a time

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2024.4

# Adjust simulation name (default filename for -deffnm option)
sim_name=md

nt=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$nt

nhours=$(squeue -h -j $SLURM_JOB_ID -O TimeLimit  cut -d: -f1)

srun gmx_mpi mdrun -ntomp $nt -deffnm $simname -cpi "$simname.cpt" -maxh $nhours

exit_code=$?
if (( exit_code != 0 )); then
    echo "Simulation exited with an error, cancelling pending jobs"
    scancel -t pending $SLURM_ARRAY_JOB_ID
    exit $exit_code
fi

nsteps_expr='^[[:space:]]*nsteps[[:space:]]*=[[:space:]]*[[:digit:]]*$'
nsteps=$(grep "$nsteps_expr" "$simname.log"  awk '{ print $3 }')
if grep "^Writing checkpoint, step $nsteps at " "$simname.log"; then
    echo "Simulation finished, cancelling pending jobs"
    scancel -t pending $SLURM_ARRAY_JOB_ID
fi
}}`{=mediawiki}`</tab>`{=html} `<tab name="GPU job">`{=html} `{{File
  |name=gromacs_job_checkpoint.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --gpus-per-node=1        # request 1 GPU per node
#SBATCH --cpus-per-task=4        # number of OpenMP threads per MPI process
#SBATCH --mem-per-cpu=2000M      # memory limit per CPU core (megabytes)
#SBATCH --time=03:00:00          # time limit (D-HH:MM:ss)
#SBATCH --array=1-20%1           # job range, running only 1 at a time

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs/2024.4

# Adjust simulation name (default filename for -deffnm option)
sim_name=md

nt=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$nt

nhours=$(squeue -h -j $SLURM_JOB_ID -O TimeLimit  cut -d: -f1)

srun gmx mdrun -ntomp $nt -deffnm $simname -cpi "$simname.cpt" -maxh $nhours

exit_code=$?
if (( exit_code != 0 )); then
    echo "Simulation exited with an error, cancelling pending jobs"
    scancel -t pending $SLURM_ARRAY_JOB_ID
    exit $exit_code
fi

nsteps_expr='^[[:space:]]*nsteps[[:space:]]*=[[:space:]]*[[:digit:]]*$'
nsteps=$(grep "$nsteps_expr" "$simname.log"  awk '{ print $3 }')
if grep "^Writing checkpoint, step $nsteps at " "$simname.log"; then
    echo "Simulation finished, cancelling pending jobs"
    scancel -t pending $SLURM_ARRAY_JOB_ID
fi
}}`{=mediawiki}`</tab>`{=html} `</tabs>`{=html}

# Performance and benchmarking {#performance_and_benchmarking}

A team at [ACENET](https://www.ace-net.ca/) has created a [Molecular Dynamics Performance Guide](https://mdbench.ace-net.ca/mdbench/) for Alliance clusters. It can help you determine optimal conditions for AMBER, GROMACS, NAMD, and OpenMM jobs. The present section focuses on GROMACS performance.

Getting the best mdrun performance with GROMACS is not a straightforward task. The GROMACS developers are maintaining a long section in their user-guide dedicated to mdrun-performance[^7] which explains all relevant options/parameters and strategies.

There is no \"One size fits all\", but the best parameters to choose highly depend on the size of the system (number of particles as well as size and shape of the simulation box) and the simulation parameters (cutoffs, use of Particle-Mesh-Ewald[^8] (PME) method for long-range electrostatics).

GROMACS prints performance information and statistics at the end of the `md.log` file, which is helpful in identifying bottlenecks. This section often contains notes on how to further improve the performance.

The **simulation performance** is typically quantified by the number of nanoseconds of MD-trajectory that can be simulated within a day (ns/day).

**Parallel scaling** is a measure of how effectively the compute resources are used. It is defined as:

:   `<span>`{=html}S = p~N~ / ( N \* p~1~ )`</span>`{=html}

Where *p~N~* is the performance using *N* CPU cores.

Ideally, the performance increases linearly with the number of CPU cores (\"linear scaling\"; S = 1).

## MPI processes / Slurm tasks / Domain decomposition {#mpi_processes_slurm_tasks_domain_decomposition}

The most straightforward way to increase the number of MPI processes (called MPI-ranks in the GROMACS documentation), which is done by using Slurm\'s `--ntasks` or `--ntasks-per-node` in the job script.

GROMACS uses **Domain Decomposition**[^9] (DD) to distribute the work of solving the non-bonded Particle-Particle (PP) interactions across multiple CPU cores. This is done by effectively cutting the simulation box along the X, Y and/or Z axes into domains and assigning each domain to one MPI process.

This works well until the time needed for communication becomes large in respect to the size (in respect of *number of particles* as well as *volume*) of the domain. In that case the parallel scaling will drop significantly below 1 and in extreme cases the performance drops when increasing the number of domains.

GROMACS can use **Dynamic Load Balancing** to shift the boundaries between domains to some extent, in order to avoid certain domains taking significantly longer to solve than others. The `mdrun` parameter `-dlb auto` is the default.

Domains cannot be smaller in any direction than the longest cutoff radius.

### Long-range interactions with PME {#long_range_interactions_with_pme}

The Particle-Mesh-Ewald method (PME) is often used to calculate the long-range non-bonded interactions (interactions beyond the cutoff radius). As PME requires global communication, the performance can degrade quickly when many MPI processes are involved that are calculating both the short-range (PP) as well as the long-range (PME) interactions. This is avoided by having dedicated MPI processes that only perform PME (PME-ranks).

GROMACS mdrun by default uses heuristics to dedicate a number of MPI processes to PME when the total number of MPI processes 12 or greater. The mdrun parameter `-npme` can be used to select the number of PME ranks manually.

In case there is a significant \"Load Imbalance\" between the PP and PME ranks (e.g. the PP ranks have more work per timestep than the PME ranks), one can shift work from the PP ranks to the PME ranks by increasing the cutoff radius. This will not affect the result, as the sum of short-range + long-range forces (or energies) will be the same for a given timestep. Mdrun will attempt to do that automatically since version 4.6 unless the mdrun parameter `-notunepme` is used.

Since version 2018, PME can be offloaded to the GPU (see below) however the implementation as of version 2018.1 has still several limitations [^10] among them that only a single GPU rank can be dedicated to PME.

## OpenMP threads / CPUs-per-task {#openmp_threads_cpus_per_task}

Once Domain Decomposition with MPI processes reaches the scaling limit (parallel scaling starts dropping), performance can be further improved by using **OpenMP threads** to spread the work of an MPI process (rank) over more than one CPU core. To use OpenMP threads, use Slurm\'s `--cpus-per-task` parameter in the job script (both for `#SBATCH>` and `srun`) and either set the *OMP_NUM_THREADS* variable with: `export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"` (recommended) or the mdrun parameter `-ntomp ${SLURM_CPUS_PER_TASK:-1}`.

According to GROMACS developers, the optimum is usually between 2 and 6 OpenMP threads per MPI process (cpus-per-task). However for jobs running on a very large number of nodes it might be worth trying even larger number of *cpus-per-task*.

Especially for systems that don\'t use PME, we don\'t have to worry about a \"PP-PME Load Imbalance\". In those cases we can choose 2 or 4 *ntasks-per-node* and set *cpus-per-task* to a value that *ntasks-per-node \* cpus-per-task* matches the number of CPU cores in a compute node.

## CPU architecture {#cpu_architecture}

GROMACS uses optimized kernel functions to compute the real-space portion of short-range, non-bonded interactions. Kernel functions are available for a variety of SIMD instruction sets, such as AVX, AVX2, and AVX512. Kernel functions are chosen when compiling GROMACS, and should match the capabilities of the CPUs that will be used to run the simulations. This is done for you by our team: when you load a GROMACS module into your environment, an appropriate AVX/AVX2/AVX512 version is chosen depending on the architecture of the cluster. GROMACS reports what SIMD instruction set it supports in its log file, and will warn you if the selected kernel function is suboptimal.

## GPUs

<div style="color: red; border: 1px dashed #2f6fab">

Tips on how to use GPUs efficiently will be added soon.

</div>

# Analyzing results {#analyzing_results}

## GROMACS tools {#gromacs_tools}

*GROMACS* contains a large number of tools that can be used for common tasks of post-processing and analysis. The *GROMACS* manual contains a [list of available commands organized by topic](https://manual.gromacs.org/current/user-guide/cmdline.html#commands-by-topic) as well as [organized by name](https://manual.gromacs.org/current/user-guide/cmdline.html#commands-by-name) that give a short description and link to the corresponding command reference.

These commands will typically read the trajectory (in the *XTC*, *TNG* or *TRR* format) as well as a coordinate file (*GRO*, *PDB*, *TPR*, etc.) and write plots in the [*XVG* format](https://manual.gromacs.org/current/reference-manual/file-formats.html#xvg) which can be used for inputs for the [plotting tool Grace](https://plasma-gate.weizmann.ac.il/Grace/) (command `xmgrace`; [Grace User Guide](https://plasma-gate.weizmann.ac.il/Grace/doc/UsersGuide.html)). As *XVG* files are simple text files, they can also be processed with scripts or imported into other spreadsheet programs.

## VMD

[VMD](https://docs.alliancecan.ca/VMD "VMD"){.wikilink} is a molecular visualization program for displaying, animating, and analyzing large biomolecular systems using 3-D graphics and built-in scripting. It can be used to visually inspect GROMACS trajectories and also offers a large number of built-in and external plugins for analysis. It can also be used in command line mode.

## Using Python {#using_python}

[MDAnalysis](https://www.mdanalysis.org/) and [MDTraj](https://www.mdtraj.org/) are two [Python](https://docs.alliancecan.ca/Python "Python"){.wikilink} packages that we provide as [precompiled Python wheels](https://docs.alliancecan.ca/Available_Python_wheels "precompiled Python wheels"){.wikilink}. They can read and write trajectory and coordinate files of *GROMACS* (*TRR* and *XTC*) and many other MD packages and also include a variety of commonly used analysis functions. *MDAnalysis* can also read topology information from *GROMACS* *TPR* files, though often not those created by the latest versions of *GROMACS*.

Both packages feature a versatile atom-selection language and expose the coordinates of the trajectories, which makes it very easy to write custom analysis tools that can be tailored to a specific problem and integrate well with Python\'s data-science packages like *NumPy*, *SciPy* and *Pandas*, as well as plotting libraries like *Matplotlib*/*Pyplot* and *Seaborn*.

# Related modules {#related_modules}

## GROMACS-Plumed {#gromacs_plumed}

PLUMED[^11] is an open source library for free energy calculations in molecular systems which works together with some of the most popular molecular dynamics engines.

The `gromacs-plumed` modules are versions of GROMACS that have been patched with PLUMED\'s modifications so that they can run meta-dynamics simulations.

`<tabs>`{=html} `<tab name="StdEnv/2023">`{=html}

  GROMACS   PLUMED   modules for running on CPUs                                  modules for running on GPUs (CUDA)                                     Notes
  --------- -------- ------------------------------------------------------------ ---------------------------------------------------------------------- -----------------------
  v2023.5   v2.9.2   `StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs-plumed/2023.5`   `StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs-plumed/2023.5`   GCC, FlexiBLAS & FFTW
  v2020.7   v2.8.5   `StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs-plumed/2020.7`   `StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs-plumed/2020.7`   GCC, FlexiBLAS & FFTW

`</tab>`{=html} `<tab name="StdEnv/2020">`{=html}

  GROMACS   PLUMED   modules for running on CPUs                                   modules for running on GPUs (CUDA)                                      Notes
  --------- -------- ------------------------------------------------------------- ----------------------------------------------------------------------- -----------------------
  v2022.6   v2.8.3   `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs-plumed/2022.6`   `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs-plumed/2022.6`   GCC, FlexiBLAS & FFTW
  v2022.3   v2.8.1   `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs-plumed/2022.3`   `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs-plumed/2022.3`   GCC, FlexiBLAS & FFTW
  v2021.6   v2.7.4   `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs-plumed/2021.6`   `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs-plumed/2021.6`   GCC, FlexiBLAS & FFTW
  v2021.4   v2.7.3   `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs-plumed/2021.4`   `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs-plumed/2021.4`   GCC, FlexiBLAS & FFTW
  v2021.2   v2.7.1   `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs-plumed/2021.2`   `StdEnv/2020 gcc/9.3.0 cuda/11.0 openmpi/4.0.3 gromacs-plumed/2021.2`   GCC & MKL
  v2019.6   v2.6.2   `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs-plumed/2019.6`   `StdEnv/2020 gcc/9.3.0 cuda/11.0 openmpi/4.0.3 gromacs-plumed/2019.6`   GCC & MKL

`</tab>`{=html} `<tab name="StdEnv/2018 and StdEnv/2016">`{=html}

  GROMACS   PLUMED   modules for running on CPUs                                        modules for running on GPUs (CUDA)                                             Notes
  --------- -------- ------------------------------------------------------------------ ------------------------------------------------------------------------------ -------------
  v2019.6   v2.5.4   `StdEnv/2018.3 gcc/7.3.0 openmpi/3.1.2 gromacs-plumed/2019.6`      `StdEnv/2018.3 gcc/7.3.0 cuda/10.0.130 openmpi/3.1.2 gromacs-plumed/2019.6`    GCC & MKL
  v2019.5   v2.5.3   `StdEnv/2018.3 gcc/7.3.0 openmpi/3.1.2 gromacs-plumed/2019.5`      `StdEnv/2018.3 gcc/7.3.0 cuda/10.0.130 openmpi/3.1.2 gromacs-plumed/2019.5`    GCC & MKL
  v2018.1   v2.4.2   `StdEnv/2016.4 gcc/6.4.0 openmpi/2.1.1 gromacs-plumed/2018.1`      `StdEnv/2016.4 gcc/6.4.0 cuda/9.0.176 openmpi/2.1.1 gromacs-plumed/2018.1`     GCC & FFTW
  v2016.3   v2.3.2   `StdEnv/2016.4 intel/2016.4 openmpi/2.1.1 gromacs-plumed/2016.3`   `StdEnv/2016.4 intel/2016.4 cuda/8.0.44 openmpi/2.1.1 gromacs-plumed/2016.3`   Intel & MKL

`</tab>`{=html} `</tabs>`{=html}

## GROMACS-Colvars {#gromacs_colvars}

Colvars[^12] is a software module for molecular simulation programs, which adds additional capabilities of collective variables to apply biasing potentials, calculate potentials-of-mean-force (PMFs) along any set of variables, use enhanced sampling methods, such as Adaptive Biasing Force (ABF), metadynamics, steered MD and umbrella sampling.

As of GROMACS v2024[^13], the Colvars library has been added to the official GROMACS releases and can be used without the need of a patched version.

Documentation on how to use Colvars with GROMACS:

- *Collective Variable simulations with the Colvars module*[^14] in the GROMACS Reference manual,
- Molecular dynamics parameters (.mdp options) for the Colvars module[^15],
- the Colvars Reference manual for GROMACS[^16],
- the publication: Fiorin et al. **2013**, *Using collective variables to drive molecular dynamics simulations.*[^17]

GROMACS versions prior to v2024 required that they have been patched with Colvars modifications, so that the collective variables can be used in simulations.\
The `gromacs-colvars/2020.6` module is such a modified version of GROMACS that includes Colvars 2021-12-20.

  GROMACS   Colvars      modules for running on CPUs                                    modules for running on GPUs (CUDA)                                       Notes
  --------- ------------ -------------------------------------------------------------- ------------------------------------------------------------------------ -----------------------
  v2020.6   2021-12-20   `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs-colvars/2020.6`   `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs-colvars/2020.6`   GCC, FlexiBLAS & FFTW

## GROMACS-CP2K {#gromacs_cp2k}

CP2K[^18] is a quantum chemistry and solid-state physics software package. Since version 2022 GROMACS can be compiled with CP2K-support[^19] to enable Hybrid Quantum-Classical simulations (QM/MM)[^20].

The `gromacs-cp2k` modules are versions of GROMACS that have been compiled with CP2K QM/MM support.

Different from other GROMACS modules, these modules are only available for CPU calculations and not for GPUs (CUDA). Also the modules contain only MPI-enabled executables:

- **`gmx_mpi`** - mixed precision GROMACS with OpenMP and MPI.
- **`gmx_mpi_d`** - double precision GROMACS with OpenMP and MPI.

  GROMACS   CP2K   modules for running on CPUs                                 Notes
  --------- ------ ----------------------------------------------------------- -----------------------
  v2022.2   9.1    `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs-cp2k/2022.2`   GCC, FlexiBLAS & FFTW

Here are links to various resources for running QM/MM simulations with this combination of GROMACS and CP2K:

- [Hybrid Quantum-Classical simulations (QM/MM) with CP2K interface](https://manual.gromacs.org/documentation/current/reference-manual/special/qmmm.html) in the GROMACS manual.
- [CP2K QM/MM Best Practices Guide](https://docs.bioexcel.eu/qmmm_bpg/) by BioExcel.
- [QM/MM with GROMACS + CP2K](https://docs.bioexcel.eu/2021-04-22-qmmm-gromacs-cp2k/) Workshop material from BioExcel.\
  This contains tutorial material for setting up and running QM/MM simulations as well as links to YouTube videos with theory lectures. This material was written to be used with HPC resources from the European Centre of Excellence for Computational Biomolecular Research (BioExcel), however only small adjustments are needed to use our HPC systems instead.\
  Most notably the command `gmx_cp2k` needs to be replaced with either `gmx_mpi` (mixed precision) or `gmx_mpi_d` (double precision) and the job scripts (which are also using Slurm), need to be adjusted as well.

:\* [GitHub Repository](https://github.com/bioexcel/2022-06-16-gromacs-cp2k-tutorial) with example file for BioExcel Tutorial.

- [GROMACS-CP2K integration](https://www.cp2k.org/tools:gromacs) on CP2K homepage.

## GROMACS-LS {#gromacs_ls}

GROMACS-LS[^21] and the MDStress library enable the calculation of local stress fields from molecular dynamics simulations. The MDStress library is included in the GROMACS-LS module.

Please refer to manual for GROMACS-LS at: [Local_stress.pdf](https://vanegaslab.org/files/Local_stress.pdf) and the publications listed therein for information about the method and how to use it.

Invoking commands like `gmx_LS mdrun -rerun` or `gmx_LS trjconv` needs a `.tpr` file. If you want to analyze a trajectory that has been simulated with a newer version of GROMACS (e.g. 2024), then an older version cannot read that .tpr file because new options are added to the format specification with every major release (2018, 2019 \... 2024). But as the answer to Q14 in the [Local_stress.pdf](https://vanegaslab.org/files/Local_stress.pdf) document suggests, you can use `gmx_LS grompp` or `gmx grompp` from the 2016.6 version (which is available as well) to create a new .tpr file using the same input files (\*.mdp, topol.top, \*.itp, \*.gro, etc.) which were used to make the `.tpr` file for the simulation. This new .tpr is then compatible with GROMACS-LS 2016.3.\
In case the `*.mdp` files used any keywords or features that were not yet present in 2016 (e.g. `pcouple = C-rescale`), then you need to either change or remove it (e.g. change to `pcouple = Berendsen`). In the case of pcouple, the result will not differ anyway, because the trajectory is processed as with the `-rerun` option and pressure coupling will not happen in that case. The mentioning of `cutoff-scheme = group` in the answer to Q14 can be ignored, because GROMACS 2016 already supports \"cutoff-scheme = Verlet\" and the \"group\" scheme was removed for GROMACS 2020. Therefore GROMACS-LS 2016.3 can be used to process simulations that used either cutoff scheme.

Notes:

- Because the manual was written for the older GROMACS-LS v4.5.5 and that the core gromacs commands have changed in version 5, you need to use commands like `gmx_LS mdrun` and `gmx_LS trjconv` instead of `mdrun_LS` and `trjconv_LS`.
- GROMACS-LS requires to be compiled in double precision does not support MPI, SIMD hardware acceleration nor GPUs and is therefore much slower than normal GROMACS. It can only use a single CPU core.
- Unlike other patched versions of GROMACS, the modules `gromacs-ls/2016.3` and `gromacs/2016.6` can be loaded at the same time.

  module              modules for running on CPUs                           Notes
  ------------------- ----------------------------------------------------- -----------------------------------------------------------------------------------
  gromacs-ls/2016.3   `StdEnv/2023 gcc/12.3 gromacs-ls/2016.3`              GROMACS-LS is a serial application and does not support MPI, OpenMP or GPUs/CUDA.
  gromacs/2016.6      `StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs/2016.6`   This Gromacs module can be used to prepare TPR input files for GROMACS-LS.

## GROMACS-RAMD {#gromacs_ramd}

GROMACS-RAMD is a fork of GROMACS that implements the *Random Acceleration Molecular Dynamics* (RAMD) method.[^22] This method can be used to identify ligand exit routes from the buried binding pockets of receptors and investigate the mechanism of ligand dissociation by running molecular dynamics simulations with an additional randomly oriented force applied to a molecule in the system.

Information on RAMD-specific MDP options[^23] can be found on the [GROMACS-RAMD GitHub page](https://github.com/HITS-MCM/gromacs-ramd).

  GROMACS   RAMD   modules for running on CPUs                                          modules for running on GPUs (CUDA)                                             Notes
  --------- ------ -------------------------------------------------------------------- ------------------------------------------------------------------------------ -----------------------
  v2024.1   2.1    `StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs-ramd/2024.1-RAMD-2.1`    `StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs-ramd/2024.1-RAMD-2.1`    GCC, FlexiBLAS & FFTW
  v2020.5   2.0    `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 gromacs-ramd/2020.5-RAMD-2.0`   `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 gromacs-ramd/2020.5-RAMD-2.0`   GCC, FlexiBLAS & FFTW

## GROMACS-SWAXS {#gromacs_swaxs}

GROMACS-SWAXS[^24] is a modified version of GROMACS for computing small- and wide-angle X-ray or neutron scattering curves (SAXS/SANS) and for doing SAXS/SANS-driven molecular dynamics simulations.

Please refer to the [GROMACS-SWAXS Documentation](https://cbjh.gitlab.io/gromacs-swaxs-docs/) for a description of the features (mdrun input and output options, mpd options, use of `gmx genscatt` and `gmx genenv` commands) that have been added in addition to normal GROMACS features, and for a number of tutorials.

  GROMACS   SWAXS   modules for running on CPUs                                       modules for running on GPUs (CUDA)                                          Notes
  --------- ------- ----------------------------------------------------------------- --------------------------------------------------------------------------- -----------------------
  v2021.7   0.5.1   `StdEnv/2023 gcc/12.3 openmpi/4.1.5 gromacs-swaxs/2021.7-0.5.1`   `StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 gromacs-swaxs/2021.7-0.5.1`   GCC, FlexiBLAS & FFTW

## G_MMPBSA

G_MMPBSA[^25] is a tool that calculates components of binding energy using MM-PBSA method except the entropic term and energetic contribution of each residue to the binding using energy decomposition scheme.

Development of that tool seems to have stalled in April 2016 and no changes have been made since then. Therefore it is only compatible with Gromacs 5.1.x. For newer version of GROMACS consider using gmx_MMPBSA[^26] instead (see below).

The version installed can be loaded with `module load StdEnv/2016.4 gcc/5.4.0 g_mmpbsa/2016-04-19` which is the most up-to-date version and consists of version 1.6 plus the change to make it compatible with Gromacs 5.1.x. The installed version has been compiled with `gromacs/5.1.5` and `apbs/1.3`.

Please be aware that G_MMPBSA uses implicit solvents and there have been studies[^27] that conclude that there are issues with the accuracy of these methods for calculating binding free energies.

## gmx_MMPBSA

gmx_MMPBSA[^28] is a tool based on [AMBER](https://docs.alliancecan.ca/AMBER "AMBER"){.wikilink}\'s MMPBSA.py aiming to perform end-state free energy calculations with GROMACS files.

Other than the older G_MMPBSA[^29], which is only compatible with older versions of GROMACS, gmx_MMPBSA can be used with current versions of GROMACS and [AmberTools](https://docs.alliancecan.ca/AMBER#AmberTools_21 "AmberTools"){.wikilink}.

Please be aware that gmx_MMPBSA uses implicit solvents and there have been studies[^30] that conclude that there are issues with the accuracy of these methods for calculating binding free energies.

### Submission scripts {#submission_scripts_1}

This submission script installs and executes gmx_MMPBSA in a temporary directory on the local disk of a compute node. All MPI tasks must be on one node. For multi-node submission install a virtual environment on a shared filesystem.

### Installing gmx_MMPBSA into a virtualenv {#installing_gmx_mmpbsa_into_a_virtualenv}

gmx_MMPBSA needs to be installed in a permanent directory if you intend to use interactive visualization.

#### Installing for gromacs/2024 (StdEnv/2023) {#installing_for_gromacs2024_stdenv2023}

1.6.3 }}

Testing.

#### Installing for gromacs/2021 (StdEnv/2020) {#installing_for_gromacs2021_stdenv2020}

1\. Load required modules and create the virtualenv 1.22.2 seaborn0.13.1 gmx_MMPBSA1.5.0.3 ParmEd3.4.4 }}

2\. The Qt/PyQt module needs to be loaded after the virtualenv is ready:

3\. Test if the main application works:

Fortunately, running the self-test is very quick, therefore it\'s permissible to run them on the login node.

Later when using gmx_MMPBSA in a job you need to load the modules and activate the virtualenv as follows:

# Links

[Biomolecular simulation](https://docs.alliancecan.ca/Biomolecular_simulation "Biomolecular simulation"){.wikilink}

- Project resources
  - Main Website: <http://www.gromacs.org/>
  - Documentation & GROMACS Manuals: <http://manual.gromacs.org/documentation/>
  - GROMACS Community Forums: <https://gromacs.bioexcel.eu/>\
    The forums are the successors to the GROMACS email lists.
- Tutorials
  - Set of 7 very good Tutorials: <http://www.mdtutorials.com/gmx/>
  - Link collection to more tutorials: <http://www.gromacs.org/Documentation/Tutorials>
- External resources
  - Tool to generate small molecule topology files: <http://www.ccpn.ac.uk/v2-software/software/ACPYPE-folder>
  - Database with Force Field topologies (CGenFF, GAFF and OPLS/AA) for small molecules: <http://www.virtualchemistry.org/>
  - Web service to generate small molecule topologies for GROMOS force fields: <https://atb.uq.edu.au/>
  - Discussion of best GPU configurations for running GROMACS: [Best bang for your buck: GPU nodes for GROMACS biomolecular simulations](https://arxiv.org/abs/1507.00898)

# References

<references />

[^1]: \"Fix missing synchronization in CUDA update kernels\" in GROMACS 2021.6 Release Notes [1](https://manual.gromacs.org/2021.6/release-notes/2021/2021.6.html#fix-missing-synchronization-in-cuda-update-kernels)

[^2]: Issue #4393 in GROMACS Project on GitLab.com [2](https://gitlab.com/gromacs/gromacs/-/issues/4393)

[^3]: \"Fix missing synchronization in CUDA update kernels\" in GROMACS 2021.6 Release Notes [3](https://manual.gromacs.org/2021.6/release-notes/2021/2021.6.html#fix-missing-synchronization-in-cuda-update-kernels)

[^4]: Issue #4393 in GROMACS Project on GitLab.com [4](https://gitlab.com/gromacs/gromacs/-/issues/4393)

[^5]: [GROMACS User Guide: Managing long simulations.](http://manual.gromacs.org/documentation/current/user-guide/managing-simulations.html)

[^6]: [GROMACS Manual page: gmx mdrun](http://manual.gromacs.org/documentation/current/onlinehelp/gmx-mdrun.html#gmx-mdrun)

[^7]: [GROMACS User-Guide: Getting good performance from mdrun](http://manual.gromacs.org/documentation/current/user-guide/mdrun-performance.html)

[^8]: [GROMACS User-Guide: Performance background information](http://manual.gromacs.org/documentation/current/user-guide/mdrun-performance.html#gromacs-background-information)

[^9]:

[^10]: [GROMACS User-Guide: GPU accelerated calculation of PME](http://manual.gromacs.org/documentation/2018.1/user-guide/mdrun-performance.html#gpu-accelerated-calculation-of-pme)

[^11]: [PLUMED Home](http://www.plumed.org/home)

[^12]: [Colvars Home](https://colvars.github.io/)

[^13]: [GROMACS 2024 Major Release Highlights](https://manual.gromacs.org/2024.1/release-notes/2024/major/highlights.html)

[^14]: [Collective Variable simulations with the Colvars module (GROMACS Reference manual)](https://manual.gromacs.org/current/reference-manual/special/colvars.html)

[^15]: [Colvars .mdp Options (GROMACS User guide)](https://manual.gromacs.org/current/user-guide/mdp-options.html#collective-variables-colvars-module)

[^16]: [Colvars Reference manual for GROMACS](https://colvars.github.io/colvars-refman-gromacs/colvars-refman-gromacs.html)

[^17]: [Fiorin et al. **2013**, *Using collective variables to drive molecular dynamics simulations.*](http://dx.doi.org/10.1080/00268976.2013.813594)

[^18]: [CP2K Home](https://www.cp2k.org/)

[^19]: [Building GROMACS with CP2K QM/MM support](https://manual.gromacs.org/documentation/current/install-guide/index.html#building-with-cp2k-qm-mm-support)

[^20]: [QM/MM with CP2K in the GROMACS Reference manual](https://manual.gromacs.org/documentation/current/reference-manual/special/qmmm.html)

[^21]: [GROMACS-LS and MDStress library](https://vanegaslab.org/software)

[^22]: [Information on the RAMD method](https://kbbox.h-its.org/toolbox/methods/molecular-simulation/random-acceleration-molecular-dynamics-ramd/)

[^23]: [RAMD-specific MDP options](https://github.com/HITS-MCM/gromacs-ramd#usage)

[^24]: [GROMACS-SWAXS Home](https://biophys.uni-saarland.de/software/gromacs-swaxs/)

[^25]: [G_MMPBSA Homepage](http://rashmikumari.github.io/g_mmpbsa/)

[^26]:

[^27]: [Comparison of Implicit and Explicit Solvent Models for the Calculation of Solvation Free Energy in Organic Solvents](http://pubs.acs.org/doi/abs/10.1021/acs.jctc.7b00169)

[^28]: [gmx_MMPBSA Homepage](https://valdes-tresanco-ms.github.io/gmx_MMPBSA/dev/getting-started/)

[^29]:

[^30]:
