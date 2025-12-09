---
title: "DL POLY/en"
url: "https://docs.alliancecan.ca/wiki/DL_POLY/en"
category: "General"
last_modified: "2025-02-20T15:28:43Z"
page_id: 11152
display_title: "DL POLY"
---

= General =

DL_POLY is a general purpose classical molecular dynamics (MD) simulation software.  It provides scalable performance from a single processor workstation to a high performance parallel computer.  DL_POLY_4 offers fully parallel I/O as well as a NetCDF alternative to the default ASCII trajectory file.

There is a mailing list here.

= License limitations =

DL_POLY is now open source and it does not require registration. A new module dl_poly4/5.1.0 is already installed under StdEnv/2023 and it is accessible for all users. However, if you would like to use the previous versions (dl_poly4/4.10.0 and/or dl_poly4/4.08), you should contact  support and ask to be added to the POSIX group that controls access to DL_POLY4. There is no need to register on DL_POLY website.

= Modules =
To see which versions of DL_POLY are installed on our systems, run module spider dl_poly4. See Using modules for more about module subcommands.

To load the version 5.x, use:

module load StdEnv/2023  intel/2023.2.1  openmpi/4.1.5 dl_poly4/5.1.0

To load the previous version 4.10.0, use:

module load StdEnv/2023 intel/2020.1.217  openmpi/4.0.3 dl_poly4/4.10.0

Note that this version requires to be added to a POSIX group as explained above in  License limitations.

We do not currently provide a module for the Java GUI interface.

= Scripts and examples =

The input files shown below (CONTROL and FIELD) were taken from example TEST01 that can be downloaded from the page of DL_POLY examples.

To start a simulation, one must have at least three files:

* CONFIG: simulation box (atomic coordinates)
* FIELD: force field parameters
* CONTROL: simulation parameters (time step, number of MD steps, simulation ensemble, ...etc.)

= Related software =

* VMD
* LAMMPS