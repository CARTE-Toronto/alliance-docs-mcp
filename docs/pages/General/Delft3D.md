---
title: "Delft3D"
url: "https://docs.alliancecan.ca/wiki/Delft3D"
category: "General"
last_modified: "2024-07-23T17:58:50Z"
page_id: 11178
display_title: "Delft3D"
---

Delft3D is a 3D modeling suite to investigate hydrodynamics, sediment transport and morphology and water quality for fluvial, estuarine and coastal environments.

= Examples =
Delft3D comes with a number of run_* scripts that are expected to be used with the Sun Grid Engine job scheduler and the MPICH library. The Alliance uses SLURM for a job scheduler and Open MPI for a default MPI implementation. To illustrate how one can run Delft3D under SLURM, we have provided submission scripts to run computational examples supplied with the software.

To copy examples into your home directory follow these steps:

$ module load StdEnv/2020  intel/2020.1.217  openmpi/4.0.3 delft3d
$ cp -a $EBROOTDELFT3D/examples ~/

Test cases within the ~/examples/ directory contain start-slurm.sh scripts that you can run with SLURM using a command such as this one:
 $ sbatch start-slurm.sh

The ~/examples/readme.examples file provides a summary of the results.