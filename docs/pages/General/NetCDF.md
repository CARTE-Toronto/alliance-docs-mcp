---
title: "NetCDF"
url: "https://docs.alliancecan.ca/wiki/NetCDF"
category: "General"
last_modified: "2026-01-21T14:03:52Z"
page_id: 5557
display_title: "NetCDF"
---

= General =

NetCDF (Network Common Data Form) is:
* an interface for array-oriented data access, and
* a library that provides an implementation of the interface.
The NetCDF library also defines a machine-independent format for representing scientific data. Together, the interface, library, and format support the creation, access, and sharing of scientific data.

There were significant changes to the library with the release of version 4.0. NetCDF 4.x is backwards compatible - it can read files produced by an earlier version but the opposite isn't true. Since 4.0 was released in 2008, we will not discuss earlier versions here.

* Project web site : https://www.unidata.ucar.edu/software/netcdf
* Manual: https://www.unidata.ucar.edu/software/netcdf/docs
* Downloads: https://www.unidata.ucar.edu/downloads/netcdf/index.jsp
* FAQ: https://www.unidata.ucar.edu/software/netcdf/docs/faq.html

== Strengths ==
* The data are independent of the processor architecture (endianness).
* The data are structured in a way that keeps track of all the pertinent information, e.g. physical units.
* NetCDF4 can read and write in parallel, if it is built using a parallel version of HDF5.
* Data can be compressed as it is written.
* A simpler interface than HDF5.
* It's free software for most platforms.

== Weak points ==
* The Python interface doesn't support parallelism (version 1.0.5)
* Certain files produced with HDF5 cannot be read using NetCDF

= Quickstart guide =
This section summarizes configuration details.

== Environment modules ==

The following modules providing NetCDF are available on our clusters via CVMFS:
* netcdf
** for linking with programs containing C instructions only
* netcdf-c++
** for linking with programs containing both C and C++ instructions
* netcdf-fortran
** for linking with programs containing Fortran instructions

There is also a set of NetCDF modules compiled with MPI support for parallel I/O:
* netcdf-mpi
** for linking with programs containing C instructions and calling MPI libraries
* netcdf-c++-mpi
** for linking with programs containing both C and C++ instructions and calling MPI
* netcdf-fortran-mpi
** for linking with programs containing Fortran instructions and calling MPI

Run module avail netcdf to see what versions are currently available with the compiler and MPI modules you have loaded. For a comprehensive list of NetCDF modules, run module -r spider '.*netcdf.*'.

Use module load netcdf/version to set your environment to use your chosen version. For example, to load the NetCDF version 4.1.3 C-based library, do:

== Submission scripts ==
Please refer to the page "Running jobs" for examples of job scripts for the Slurm workload manager. We recommend you include the module load ... command in your job script.

== Linking code to NetCDF libraries ==
Below are a few examples showing how to link NetCDF libraries to C and Fortran code:

=== Serial NetCDF ===
For C code,

For Fortran code, notice that two libraries are required and their order matters:

=== Parallel NetCDF ===
For C code which calls MPI,

=== Example ===
An example demonstrating the use of the NetCDF can be found at the following link. The example program creates a NetCDF file containing a single two-dimensional variable called "data", and labels the two dimensions "x" and "y".

To compile the example:

== NetCDF utilities ==
There are several utilities that read or write files in different formats.
* ncdump
: "The ncdump tool generates the CDL text representation of a netCDF dataset on standard output, optionally excluding some or all of the variable data in the output. The output from ncdump is intended to be acceptable as input to ncgen. Thus ncdump and ncgen can be used as inverses to transform data representation between binary and text representation." See ncdump on the UCAR site for more.
* ncgen
: This utility takes an input file in CDL format and creates a binary netCDF file. It is the reverse of ncdump. See ncgen for more.
*nccopy
: Copies a netCDF file and can change binary format, chunk sizes, compression, and other storage settings. See nccopy for more.

There are also utilities nc-config and nf-config to help you find the correct compilation and linking commands. More on them can be found here.

= PnetCDF =

PnetCDF is another library for reading and writing NetCDF-format files. It uses different procedure names than NetCDF and provides non-blocking procedures. See the PnetCDF web site for a detailed discussion.