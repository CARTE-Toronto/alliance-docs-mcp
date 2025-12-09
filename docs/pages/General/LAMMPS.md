---
title: "LAMMPS/en"
url: "https://docs.alliancecan.ca/wiki/LAMMPS/en"
category: "General"
last_modified: "2025-03-06T14:09:16Z"
page_id: 8890
display_title: "LAMMPS"
---

Parent page: Biomolecular simulation

= General =

LAMMPS is a classical molecular dynamics code. The name stands for Large-scale Atomic / Molecular Massively Parallel Simulator. LAMMPS is distributed by  Sandia National Laboratories, a US Department of Energy laboratory.

* Project web site: http://lammps.sandia.gov/
* Documentation: Online Manual.
* Mailing List: http://lammps.sandia.gov/mail.html

LAMMPS is parallelized with MPI and OpenMP, and can run on GPUs.

= Force fields =

All supported force fields are listed on the package web site,
classified by functional form (e.g. pairwise potentials, many-body potentials, etc.)
The large number of supported force fields makes LAMMPS suitable for many areas of application.
Here are some types of modelling and force fields suitable for each:

* Biomolecules: CHARMM, AMBER, OPLS, COMPASS (class 2), long-range Coulombics via PPPM, point dipoles, ...
* Polymers: all-atom, united-atom, coarse-grain (bead-spring FENE), bond-breaking, …
* Materials: EAM and MEAM for metals, Buckingham, Morse, Yukawa, Stillinger-Weber, Tersoff, EDIP, COMB, SNAP, ...
* Reactions: AI-REBO, REBO, ReaxFF, eFF
* Mesoscale: granular, DPD, Gay-Berne, colloidal, peri-dynamics, DSMC...

Combinations of potentials can be used for hybrid systems, e.g. water on metal, polymer/semiconductor interfaces, colloids in solution, ...

= Versions and packages =

To see which versions of LAMMPS are installed on our systems, run module spider lammps. See Using modules for more about module subcommands.

LAMMPS version numbers are based on their release dates, and have the format YYYYMMDD.  You should run:

 module avail lammps

to see all the releases that are installed, so you can find the one which is most appropriate for you to use.

For each release installed, one or more modules are are available.
For example, the release of 31 March 2017 has three modules:

* Built with MPI: lammps/20170331
* Built with USER-OMP support:  lammps-omp/20170331
* Built with USER-INTEL support: lammps-user-intel/20170331

These versions are also available with GPU support.
In order to use the GPU-enabled version, load the CUDA module before loading the LAMMPS module:

 $ module load cuda
 $ module load lammps-omp/20170331

The name of the executable may differ from one version to another. All prebuilt versions on our clusters have a symbolic link called lmp. It means that no matter which module you pick, you can execute LAMMPS by calling lmp.

If you wish to see the original name of the executable for a given module, list the files in the ${EBROOTLAMMPS}/bin directory. For example:

 $ module load lammps-omp/20170331
 $ ls ${EBROOTLAMMPS}/bin/
 lmp lmp_icc_openmpi

In this example the executable is lmp_icc_openmpi, and lmp is the symbolic link to it.

The reason there are different modules for the same release is the difference in the packages included. Recent versions of LAMMPS contain about 60 different packages that can be enabled or disabled when compiling the program. Not all packages can be enabled in a single executable. All packages are documented on the official web page.  If your simulation does not work with one module, it may be related to the fact that a necessary package was not enabled.

For some LAMMPS modules we provide a file list-packages.txt listing the enabled ("Supported") and disabled ("Not Supported") packages. Once you have loaded a particular module, run cat ${EBROOTLAMMPS}/list-packages.txt to see the contents.

If list-packages.txt is not found, you may be able to determine which packages are available by examining the EasyBuild recipe file, $EBROOTLAMMPS/easybuild/LAMMPS*.eb.  The list of enabled packages will appear in the block labelled general_packages.

= Example of input file =

The input file below can be used with either of the example job scripts.

= Performance =

Most of the CPU time for molecular dynamics simulations is spent in computing the pair interactions between particles. LAMMPS uses domain decomposition to split the work among the available processors by assigning a part of the simulation box to each processor. During the computation of the interactions between particles, communication between the processors is required. For a given number of particles, the more processors that are used, the more parts of the simulation box there are which must exchange data. Therefore, the communication time increases with increasing number of processors, eventually leading to low CPU efficiency.

Before running extensive simulations for a given problem size or a size of the simulation box, you should run tests to see how the program's performance changes with the number of cores. Run short tests using different numbers of cores to find a suitable number of cores that will (approximately) maximize the efficiency of the simulation.

The following example shows the timing for a simulation of a system of 4000 particles using 12 MPI tasks. This is an example of a very low efficiency: by using 12 cores, the system of 4000 atoms was divided to 12 small boxes. The code spent 46.45% of the time computing pair interactions and 44.5% in communications between the processors. The large number of small boxes for a such small system is responsible for the large fraction of time spent in communication.

Loop time of 15.4965 on 12 procs for 25000 steps with 4000 atoms.
Performance: 696931.853 tau/day, 1613.268 timesteps/s.
90.2% CPU use with 12 MPI tasks x 1 OpenMP threads.	Loop time of 15.4965 on 12 procs for 25000 steps with 4000 atoms.
Performance: 696931.853 tau/day, 1613.268 timesteps/s.
90.2% CPU use with 12 MPI tasks x 1 OpenMP threads.	Loop time of 15.4965 on 12 procs for 25000 steps with 4000 atoms.
Performance: 696931.853 tau/day, 1613.268 timesteps/s.
90.2% CPU use with 12 MPI tasks x 1 OpenMP threads.	Loop time of 15.4965 on 12 procs for 25000 steps with 4000 atoms.
Performance: 696931.853 tau/day, 1613.268 timesteps/s.
90.2% CPU use with 12 MPI tasks x 1 OpenMP threads.	Loop time of 15.4965 on 12 procs for 25000 steps with 4000 atoms.
Performance: 696931.853 tau/day, 1613.268 timesteps/s.
90.2% CPU use with 12 MPI tasks x 1 OpenMP threads.	Loop time of 15.4965 on 12 procs for 25000 steps with 4000 atoms.
Performance: 696931.853 tau/day, 1613.268 timesteps/s.
90.2% CPU use with 12 MPI tasks x 1 OpenMP threads.
Section	min time	avg time	max time	%varavg	%total
Pair   	6.6964  	7.1974  	7.9599  	14.8   	46.45
Neigh  	0.94857 	1.0047  	1.0788  	4.3    	6.48
Comm   	6.0595  	6.8957  	7.4611  	17.1   	44.50
Output 	0.01517 	0.01589 	0.019863	1.0    	0.10
Modify 	0.14023 	0.14968 	0.16127 	1.7    	0.97
Other  	--      	0.2332  	--      	--     	1.50

In the next example, we compare the time spent in communication and computing the pair interactions for different system sizes:

     	2048 atoms	2048 atoms	4000 atoms	4000 atoms	6912 atoms	6912 atoms	13500 atoms	13500 atoms
Cores	Pairs     	Comm      	Pairs     	Comm      	Pairs     	Comm      	Pairs      	Comm
1    	73.68     	1.36      	73.70     	1.28      	73.66     	1.27      	73.72      	1.29
2    	70.35     	5.19      	70.77     	4.68      	70.51     	5.11      	67.80      	8.77
4    	62.77     	13.98     	64.93     	12.19     	67.52     	8.99      	67.74      	8.71
8    	58.36     	20.14     	61.78     	15.58     	64.10     	12.86     	62.06      	8.71
16   	56.69     	20.18     	56.70     	20.18     	56.97     	19.80     	56.41      	20.38