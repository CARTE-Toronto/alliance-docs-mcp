---
title: "AMBER/en"
url: "https://docs.alliancecan.ca/wiki/AMBER/en"
category: "General"
last_modified: "2025-09-09T20:12:01Z"
page_id: 4211
display_title: "AMBER"
---

`<languages />`{=html}

## Introduction

[Amber](https://ambermd.org/) is the collective name for a suite of programs that allow users to perform molecular dynamics simulations, particularly on biomolecules. None of the individual programs carry this name, but the various parts work reasonably well together, and provide a powerful framework for many common calculations.

## Amber modules {#amber_modules}

We provide modules for Amber, AmberTools, and Amber-PMEMD in our [software stack](https://docs.alliancecan.ca/Available_software "software stack"){.wikilink}.

- **[AmberTools](https://ambermd.org/AmberTools.php)** (module `ambertools`) - Tools for preparing/analyzing simulations, `QUICK` for GPU-accelerated DFT calculations and `sander` for molecular dynamics. Free and open source.
- **[Amber](https://ambermd.org/AmberMD.php)** (module `amber`) - Everything included in AmberTools, plus the advanced \`pmemd\` program for high-performance molecular dynamics simulations.
- **Amber-PMEMD** (module `amber-pmemd`, Amber 24+) -- High-performance MD engine `pmemd`, optimized for CPU and GPU.\
  Provides the high-performance MD engine `pmemd` (optimized for CPU/GPU) as a **standalone module**. This change was made because starting with Amber 24, `pmemd` no longer requires AmberTools for compilation.\
  **Note:** The `amber-pmemd` module does not include AmberTools. To use both, load the `ambertools` module as well.

To see a list of installed versions and which other modules they depend on, you can use the `module spider` [command](https://docs.alliancecan.ca/Using_modules#Sub-command_spider "command"){.wikilink} or check the [Available software](https://docs.alliancecan.ca/Available_software "Available software"){.wikilink} page.

## Using AMBER on H100 GPU Clusters {#using_amber_on_h100_gpu_clusters}

**Key Update:** Older AMBER modules are incompatible with NVIDIA H100 GPUs. For GPU-accelerated runs, use the newly installed modules below.

### Module Requirements: {#module_requirements}

`ambertools/25.0` or `amber-pmemd/24.3`

These modules include H100-specific CUDA kernels (compiled with CUDA 12+ for the Hopper architecture).

**Important:** Do not use legacy AMBER modules for GPU jobs --- they will fail on H100 nodes.

## Loading modules {#loading_modules}

`<tabs>`{=html} `<tab name="StdEnv/2023">`{=html}

  AMBER version      modules for running on CPUs                                       modules for running on GPUs (CUDA)                                Notes
  ------------------ ----------------------------------------------------------------- ----------------------------------------------------------------- ------------------------------------
  amber-pmemd/24.3   `StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.6 amber-pmemd/24.3`   `StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.6 amber-pmemd/24.3`   H100 compatible
  amber/22.5-23.5    `StdEnv/2023 gcc/12.3 openmpi/4.1.5 amber/22.5-23.5`              `StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 amber/22.5-23.5`    
  ambertools/25.0    `StdEnv/2023 gcc/12.3 openmpi/4.1.5 ambertools/25.0`              `StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.6 ambertools/25.0`    H100 compatible, with PLUMED/2.9.0
  ambertools/23.5    `StdEnv/2023 gcc/12.3 openmpi/4.1.5 ambertools/23.5`              `StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 ambertools/23.5`    
                                                                                                                                                         

`</tab>`{=html} `<tab name="StdEnv/2020">`{=html}

  AMBER version       modules for running on CPUs                                       modules for running on GPUs (CUDA)                                          Notes
  ------------------- ----------------------------------------------------------------- --------------------------------------------------------------------------- -----------------------
  ambertools/21       `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 scipy-stack ambertools/21`   `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 scipy-stack ambertools/21`   GCC, FlexiBLAS & FFTW
  amber/20.12-20.15   `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 amber/20.12-20.15`           `StdEnv/2020 gcc/9.3.0 cuda/11.4 openmpi/4.0.3 amber/20.12-20.15`           GCC, FlexiBLAS & FFTW
  amber/20.9-20.15    `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 amber/20.9-20.15`            `StdEnv/2020 gcc/9.3.0 cuda/11.0 openmpi/4.0.3 amber/20.9-20.15`            GCC, MKL & FFTW
  amber/18.14-18.17   `StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 amber/18.14-18.17`           `StdEnv/2020 gcc/8.4.0 cuda/10.2 openmpi/4.0.3`                             GCC, MKL
                                                                                                                                                                    

`</tab>`{=html} `<tab name="StdEnv/2016">`{=html}

  AMBER version       modules for running on CPUs                                                 modules for running on GPUs (CUDA)                                                       Notes
  ------------------- --------------------------------------------------------------------------- ---------------------------------------------------------------------------------------- ----------------------------------------------------------------------
  amber/18            `StdEnv/2016 gcc/5.4.0 openmpi/2.1.1 scipy-stack/2019a amber/18`            `StdEnv/2016 gcc/5.4.0 openmpi/2.1.1 cuda/9.0.176 scipy-stack/2019a amber/18`            GCC, MKL
  amber/18.10-18.11   `StdEnv/2016 gcc/5.4.0 openmpi/2.1.1 scipy-stack/2019a amber/18.10-18.11`   `StdEnv/2016 gcc/5.4.0 openmpi/2.1.1 cuda/9.0.176 scipy-stack/2019a amber/18.10-18.11`   GCC, MKL
  amber/18.10-18.11   `StdEnv/2016 gcc/7.3.0 openmpi/3.1.2 scipy-stack/2019a amber/18.10-18.11`   `StdEnv/2016 gcc/7.3.0 cuda/9.2.148 openmpi/3.1.2 scipy-stack/2019a amber/18.10-18.11`   GCC, MKL
  amber/16            `StdEnv/2016.4 amber/16`                                                                                                                                             Available only on Graham. Some Python functionality is not supported

`</tab>`{=html}

`</tabs>`{=html}

## Using modules {#using_modules}

### AmberTools 21 {#ambertools_21}

Currently, AmberTools 21 module is available on all clusters. AmberTools provide the following MD engines: sander, sander.LES, sander.LES.MPI, sander.MPI, sander.OMP, sander.quick.cuda, and sander.quick.cuda.MPI. After loading the module set AMBER environment variables:

`source $EBROOTAMBERTOOLS/amber.sh`

### Amber 20 {#amber_20}

There are two versions of amber/20 modules: 20.9-20.15 and 20.12-20.15. The first one uses MKL and cuda/11.0, while the second uses FlexiBLAS and cuda/11.4. MKL libraries do not perform well on AMD CPU, and FlexiBLAS solves this problem. It detects CPU type and uses libraries optimized for the hardware. cuda/11.4 is required for running simulations on A100 GPUs installed on Narval.

CPU-only modules provide all MD programs available in AmberTools/20 plus pmemd (serial) and pmemd.MPI (parallel). GPU modules add pmemd.cuda (single GPU), and pmemd.cuda.MPI (multi - GPU).

### Known issues {#known_issues}

1\. Module amber/20.12-20.15 does not have MMPBSA.py.MPI executable.

2\. MMPBSA.py from amber/18-10-18.11 and amber/18.14-18.17 modules cannot perform PB calculations. Use more recent amber/20 modules for this type of calculations.

## Job submission examples {#job_submission_examples}

### Single GPU job {#single_gpu_job}

For GPU-accelerated simulations on Narval, use amber/20.12-20.15. Modules compiled with CUDA version \< 11.4 do not work on A100 GPUs. Below is an example submission script for a single-GPU job.

### CPU-only parallel MPI job {#cpu_only_parallel_mpi_job}

`<tabs>`{=html} `<tab name="Narval">`{=html} `</tab>`{=html} `<tab name="Rorqual">`{=html} `</tab>`{=html} `<tab name="Fir">`{=html} `</tab>`{=html} `<tab name="Nibi">`{=html} `</tab>`{=html} `<tab name="Trillium">`{=html} `</tab>`{=html} `</tabs>`{=html}

### QM/MM distributed multi-GPU job {#qmmm_distributed_multi_gpu_job}

The example below requests eight GPUs.

### Parallel MMPBSA job {#parallel_mmpbsa_job}

The example below uses 32 MPI processes. MMPBSA scales linearly because each trajectory frame is processed independently.

You can modify scripts to fit your simulation requirements for computing resources. See [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink} for more details.

## Performance and benchmarking {#performance_and_benchmarking}

A team at [ACENET](https://www.ace-net.ca/) has created a [Molecular Dynamics Performance Guide](https://mdbench.ace-net.ca/mdbench/) for Alliance clusters. It can help you determine optimal conditions for AMBER, GROMACS, NAMD, and OpenMM jobs. The present section focuses on AMBER performance.

View benchmarks of simulations with PMEMD[1](http://mdbench.ace-net.ca/mdbench/bform/?software_contains=PMEMD&software_id=&module_contains=&module_version=&site_contains=&gpu_model=&cpu_model=&arch=&dataset=6n4o)

View benchmarks of QM/MM simulations with SANDER.QUICK [2](http://mdbench.ace-net.ca/mdbench/bform/?software_contains=&software_id=&module_contains=&module_version=&site_contains=&gpu_model=&cpu_model=&arch=&dataset=4cg1).
