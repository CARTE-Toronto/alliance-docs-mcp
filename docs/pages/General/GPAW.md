---
title: "GPAW/en"
url: "https://docs.alliancecan.ca/wiki/GPAW/en"
category: "General"
last_modified: "2023-04-21T20:20:50Z"
page_id: 21749
display_title: "GPAW"
---

`<languages />`{=html}

\_\_TOC\_\_

# General

[GPAW](https://wiki.fysik.dtu.dk/gpaw/) is a density-functional theory (DFT) [Python](https://docs.alliancecan.ca/Python "Python"){.wikilink} code based on the projector-augmented wave (PAW) method and the atomic simulation environment (ASE).

# Creating a GPAW virtual environment {#creating_a_gpaw_virtual_environment}

We provide precompiled [Python wheels](https://docs.alliancecan.ca/Available_Python_wheels "Python wheels"){.wikilink} for GPAW that can be installed into a [virtual python environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual python environment"){.wikilink}.

1\. Check which versions of gpaw are available:

2\. Load a Python module (e.g. python/3.10)

3\. Create a new virtualenv

4\. Activate the virtualenv (venv)

5\. Install gpaw into venv

6\. Download the data and install it into the SCRATCH filesystem

7\. Now set GPAW_SETUP_PATH to point to the data directory \$SCRATCH/gpaw-setups-0.9.20000 }}

8\. We can run the tests, which are very fast:

`python-3.10.2     /home/name/venv_gpaw/bin/python                                                         `\
`gpaw-22.8.0       /home/name/venv_gpaw/lib/python3.10/site-packages/gpaw/                                 `\
`ase-3.22.1        /home/name/venv_gpaw/lib/python3.10/site-packages/ase/                                  `\
`numpy-1.23.0      /home/name/venv_gpaw/lib/python3.10/site-packages/numpy/                                `\
`scipy-1.9.3       /home/name/venv_gpaw/lib/python3.10/site-packages/scipy/                                `\
`libxc-5.2.3       yes                                                                                     `\
`_gpaw             /home/name/venv_gpaw/lib/python3.10/site-packages/_gpaw.cpython-310-x86_64-linux-gnu.so `\
`MPI enabled       yes                                                                                     `\
`OpenMP enabled    yes                                                                                     `\
`scalapack         yes                                                                                     `\
`Elpa              no                                                                                      `\
`FFTW              yes                                                                                     `\
`libvdwxc          no                                                                                      `\
`PAW-datasets (1)  /scratch/name/gpaw-setups-0.9.20000                                                     `\
`-----------------------------------------------------------------------------------------------------------`

Doing a test calculation (cores: 1): \... Done Test parallel calculation with \"gpaw -P 4 test\". }}

`python-3.10.2     /home/name/venv_gpaw/bin/python                                                         `\
`gpaw-22.8.0       /home/name/venv_gpaw/lib/python3.10/site-packages/gpaw/                                 `\
`ase-3.22.1        /home/name/venv_gpaw/lib/python3.10/site-packages/ase/                                  `\
`numpy-1.23.0      /home/name/venv_gpaw/lib/python3.10/site-packages/numpy/                                `\
`scipy-1.9.3       /home/name/venv_gpaw/lib/python3.10/site-packages/scipy/                                `\
`libxc-5.2.3       yes                                                                                     `\
`_gpaw             /home/name/venv_gpaw/lib/python3.10/site-packages/_gpaw.cpython-310-x86_64-linux-gnu.so `\
`MPI enabled       yes                                                                                     `\
`OpenMP enabled    yes                                                                                     `\
`scalapack         yes                                                                                     `\
`Elpa              no                                                                                      `\
`FFTW              yes                                                                                     `\
`libvdwxc          no                                                                                      `\
`PAW-datasets (1)  /scratch/name/gpaw-setups-0.9.20000                                                     `\
`-----------------------------------------------------------------------------------------------------------`

Doing a test calculation (cores: 4): \... Done }}

Results of the last test can be found in the file `test.txt` that will be created in the current directory.

# Example Jobscript {#example_jobscript}

A jobscript may look something like this for hybrid (OpenMP and MPI) parallelization. This assumes that the virtualenv is in your \$HOME directory and the PAW-datasets in \$SCRATCH as shown above. `{{File
|language=bash
|name=job_gpaw.sh
|contents=
#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=0-01:00
module load gcc/9.3.0 openmpi/4.0.3
source ~/venv_gpaw/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export GPAW_SETUP_PATH=/scratch/$USER/gpaw-setups-0.9.20000

srun --cpus-per-task=$OMP_NUM_THREADS gpaw python my_gpaw_script.py
}}`{=mediawiki} This would use a single node with 8 MPI-ranks (ntasks) and 4 OpenMP threads per MPI rank (cpus-per-task) so a total of 32 CPUs. You probably want to adjust those numbers so that the product matches the number of cores of a whole node (i.e. 32 at [Graham](https://docs.alliancecan.ca/Graham "Graham"){.wikilink}, 40 at [Béluga](https://docs.alliancecan.ca/Béluga/en "Béluga"){.wikilink} and [Niagara](https://docs.alliancecan.ca/Niagara "Niagara"){.wikilink}, 48 at [Cedar](https://docs.alliancecan.ca/Cedar "Cedar"){.wikilink} or 64 at [Narval](https://docs.alliancecan.ca/Narval/en "Narval"){.wikilink}).

Setting `OMP_NUM_THREADS` as shown above, makes sure it is always set to the same value as cpus-per-task or 1 in case cpus-per-task is not set. Loading the modules `gcc/9.3.0` and `openmpi/4.0.3` ensures that the exact MPI library is used for the job, as was used for building the wheels.
