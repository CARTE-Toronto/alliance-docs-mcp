---
title: "CP2K/en"
url: "https://docs.alliancecan.ca/wiki/CP2K/en"
category: "General"
last_modified: "2023-04-25T15:53:28Z"
page_id: 17029
display_title: "CP2K"
---

CP2K is a quantum chemistry and solid state physics software package that can perform atomistic simulations of solid state, liquid, molecular, periodic, material, crystal, and biological systems.

== Versions ==

The latest version installed is CP2K 8.2. You can load the module compiled with GCC using

 module load StdEnv/2020  gcc/9.3.0  openmpi/4.0.3 cp2k/8.2

You can also choose to use the version compiled with the Intel compiler if you prefer, but it seems less stable, as it sometimes crashes for unknown reasons.

 module load StdEnv/2020  intel/2020.1.217  openmpi/4.0.3 cp2k/8.2

== Example job ==

Here we will use the static calculation example from the CP2K website

First, log into one of our clusters and download the needed files with the following commands:

 wget https://www.cp2k.org/_media/static_calculation.tgz
 tar xvfz static_calculation.tgz
 cd static_calculation/sample_output_no_smearing

Then, in that directory, create the following job submission script, with the account name changed to the one you are using.

To submit this job, execute:

 sbatch mpi_job.sh

To see if the job completed, run the command

 sq

If your job is no longer listed, that means it has completed.

The output of CP2K will be located in the file Si_bulk8.out.  There will also be an output file named slurm-*.out which should be empty if the calculation completed without error.

== Threaded/MPI jobs ==

The installation of CP2K version 8.2 and later includes both the MPI executable cp2k.popt and the OpenMP/MPI executable cp2k.psmp, which may give better performance for some calculations.  Our test shows a 10% performance increase for QS/H2O-512.inp benchmark when using 2 threads
per MPI process, compared to running MPI-only executable cp2k.popt (both runs used the same number of CPU cores in total).

Below is an example OpenMP/MPI job submission file for the Beluga cluster.  To use on other clusters, the number of tasks would have to be adjusted to match the number of cores available on the nodes of that cluster.  The performance changes when threads are used is highly problem dependent, and running the cp2k.psmp executable may actually be slower for some cases, so you must benchmark your code and choose the right option for your particular case.