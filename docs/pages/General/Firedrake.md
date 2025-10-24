---
title: "Firedrake"
url: "https://docs.alliancecan.ca/wiki/Firedrake"
category: "General"
last_modified: "2025-07-31T19:23:02Z"
page_id: 29047
display_title: "Firedrake"
---

[Firedrake](https://www.firedrakeproject.org/) is an automated system for the solution of partial differential equations using the finite element method (FEM).

Please note that every release of Firedrake requires a specific version of PETSc and several other modules or Python wheels.

# Installation

Please note that all modules must be loaded before creating and/or activating the Python virtualenv.

## Firedrake 2025.4.2 {#firedrake_2025.4.2}

2025.4.2 }}

The above has been tested with both `python/3.13` as well as `python/3.12`.

# Running jobs {#running_jobs}

## Firedrake 2025.4.2 {#firedrake_2025.4.2_1}

```{=mediawiki}
{{File
  |name=job_firedrake_multigrid.sh
  |lang="bash"
  |contents=
#!/bin/bash
#SBATCH --time=0-00:15:00  # d-hh:mm:ss
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4000M
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Load modules
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 
module load python/3.13 mpi4py/4.0.3 
module load symengine/0.14.0 libspatialindex/1.9.3 petsc/3.23.4

# activate virtualenv
source venv-firedrake/bin/activate

# run MPI job
# example from: https://firedrakeproject.org/demos/geometric_multigrid.py
srun  python  geometric_multigrid.py

}}
```
The above has been tested with both `python/3.13` as well as `python/3.12`.

# Optional dependencies {#optional_dependencies}

Firedrake has a number of [optional dependencies](https://www.firedrakeproject.org/install.html#optional-dependencies) that can be installed into the virtualenv:

- **SLEPc** and slepc4py are part of the petsc module and always available.
- **netgen**: we provide [precompiled wheels](https://docs.alliancecan.ca/Python#Available_wheels "precompiled wheels"){.wikilink} for `ngsPETSc` and `netgen_mesher`.
- **PyTorch**: since we provide precompiled [precompiled wheels](https://docs.alliancecan.ca/Python#Available_wheels "precompiled wheels"){.wikilink} for `torch`.
- **Jax**: since we provide precompiled [precompiled wheels](https://docs.alliancecan.ca/Python#Available_wheels "precompiled wheels"){.wikilink} for `jax`.
- **VTK**: currently we don\'t have a module for VTK that supports recent enough versions of Python for Firedrake (Python 3.12 and newer).\
  As a temporary workaround VTK can be installed into the virtualenv with: `pip install --no-index --find-links ~stuekero/wheels/vtk vtk==9.4.2` until we install a new VTK module.
