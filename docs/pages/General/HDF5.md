---
title: "HDF5/en"
url: "https://docs.alliancecan.ca/wiki/HDF5/en"
category: "General"
last_modified: "2025-08-29T13:27:02Z"
page_id: 6595
display_title: "HDF5"
---

= General =

HDF5 (Hierarchical Data Format) is a library for managing the formatting of scientific data. It allows you to store, read, visualize, manipulate and analyze the data in an efficient manner.
It supports an unlimited variety of datatypes, and is designed for flexible and efficient I/O and for high volume and complex data. HDF5 is portable and is extensible, allowing applications to evolve in their use of HDF5. The HDF5 Technology suite includes tools and applications for managing, manipulating, viewing, and analyzing data in the HDF5 format.
HDF (also known as HDF4) is a library and multi-object file format for storing and managing data between machines. There are two versions of HDF: HDF4 and HDF5. HDF4 is the first HDF format. Although HDF4 is still funded, new users that are not constrained to using HDF4, should use HDF5.

== Description ==
HDF was  designed for:
* high volume and/or complex data (but can be used for low volume/simple data)
* every size and type of system (portable)
* flexible, efficient storage and I/O
* applications to evolve in their use of HDF5 and to accommodate new models

HDF consists of:
* a file format for storing HDF4/HDF5 data.
* a model for organizing and accessing HDF4/HDF5 data from various applications.
* lots of software including libraries, language interfaces and many useful tools for working with this format.

List of useful links:
* Project web site : https://www.hdfgroup.org/solutions/hdf5/
* Manual: https://support.hdfgroup.org/documentation/
* Downloads: https://www.hdfgroup.org/downloads/hdf5

== Strengths ==
* The data are independent of the processor architecture (endianness).
* The data are structured in such a way as to keep all the pertinent information (e.g. physical units).
* It can be used in parallel (MPI-IO).
* It is possible to compress the data when upon writing (zlib or szip).
* It has interfaces for C, C++, Fortran 90, Java and Python.
* It can manage any sort of data (more options than NetCDF).
* It is able to read and write the .mat format used by Matlab.
* It is free software for most platforms.

== Weak points ==
* The interface is more complicated than that of NetCDF.
* HDF5 does not enforce the use of UTF-8, so the ASCII is expected most of the time.
* Dataset can not be freed in a file without first creating a file copy with the use of some external tools.

= Quickstart Guide =
This section summarizes configuration details.

== Environment Modules ==
The following modules providing HDF are available on our clusters:
* hdf
** contains HDF of version 4.1 and previous releases
* hdf5
** contains the most recent version of HDF5
*hdf5-mpi
** includes support of MPI

Run module avail hdf to see what versions are currently available with the compiler and MPI modules you have loaded. For a comprehensive list of HDF4/HDF5 modules, run module -r spider '.*hdf.*'.

Use module load hdf/version (or module load hdf5/version) to set your environment to use your chosen version.  For example, to load the HDF5 version 1.14.2 library do:

== Submission scripts ==

Please refer to the page "Running jobs" for examples of job scripts for the Slurm workload manager. We recommend you include the module load ... command in your job script.

== Linking code to HDF libraries ==
Below are few example showing how to link the HDF5 libraries in serial and parallel mode:

=== Serial HDF5 ===
Using serial library hdf5

=== Parallel HDF5 ===
Using parallel (MPI) library hdf5-mpi

=== Example ===
Th example demonstrating the use of the HDF5 library can be found at the following link. The example program reads and writes data to a dataset. It first writes integers to a dataset with data space dimensions of DIM0xDIM1, then closes the file. It then reopens the file, reads back the data, and outputs it to the screen.

To compile and run the example:

== HDF5 utilities ==
There are several helpful utilities that interface with the HDF5 libraries. The bellow list is partial. For a complete list of HDF5 utilities see this link on the Hdfgroup website.
* HDF5 ODBC Connector
: SQL-based interface for querying HDF5 data in Excel, Tableau and other tools
* HDFView
: HDF Java browser and Object Package for HDF5-1.10 (64-bit object identifiers) and HDF 4.2.12 (and above)
* HDF5 Command-line Tools (some of them)
** gif2h5/h52gif
** h5cc, h5fc, h5c++
** h5debug
** h5diff
** h5dump
** h5import
* h5check
: a tool to check the validity of the HDF5 file
* h5edit
: a tool for editing an HDF5 file