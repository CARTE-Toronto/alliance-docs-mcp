---
title: "Vtune/en"
url: "https://docs.alliancecan.ca/wiki/Vtune/en"
category: "General"
last_modified: "2024-02-14T15:15:05Z"
page_id: 15889
display_title: "Vtune"
---

__FORCETOC__

= Introduction =

VTune is Intel's Performance Analysis tool for applications and systems. It is capable of Analyzing both OpenMP and MPI based applications.

= Software module =

To load the module on any Alliance cluster run:

= Tool renaming =

The content of this page is largely concerned with the legacy version named Intel® VTune™ Amplifier.  Please note this tool has been renamed throughout Intel's documentation in latest versions (newer than the latest vtune module versions presently available on Alliance clusters) from Intel® VTune™ Amplifier to Intel® VTune™ Profiler.  Likewise the application commands amplxe-cl and amplxe-gui have been renamed to vtune and vtune-gui for both the command line and graphical tools respectively.  Further information can be found here.

= Analysis types =

To collect analysis information run:

where  should be replaced by one of the available analysis, e.g. hotspots, and  is the path to the executable you would like to analyze.  It is recommended to compile your executable with the "-g" option and to use the same optimization level as normal so as to obtain accurate results.  A listing of version specific argument options and several usage examples maybe displayed on the command line by running vtune -help, after loading the vtune module.  Complete downloadable documentation for Parallel Studio XE (including VTune) for all recent versions can be found here.  The latest version of the Intel VTune Profiler User Guide may be found here.

= Create reports =

To create a report run this command:

where  is the type of the report to generate, e.g. hotspots.  See also:
* https://software.intel.com/en-us/vtune-amplifier-help-generating-command-line-reports

= Matrix example =

Analyze and generate a summary report for the Intel Matrix Sample Project run from the command line with 4 cores:

 salloc --time=1:00:00 --cpus-per-task=4 --ntasks=1 --mem=16G --account=def-yours
 module load StdEnv/2020 vtune
 cp -a $EBROOTVTUNE/vtune/$EBVERSIONVTUNE*/samples/en/C++/matrix . cd matrix/linux
 make icc
 vtune -collect hotspots ../matrix
 vtune -report summary

The latest version of matrix_multiply (uses cmake to build) can be found here.

= Graphical mode =

The Intel Matrix Sample Project can also be run using Vtune in GUI mode as explored here .  To run VTune over VNC follow the below directions depending on which system you wish to use.  Running VTune graphically can be useful to generate command line configurations as discussed in .

== Cluster nodes ==

# Connect to a cluster compute or login node with TigerVNC
# module load StdEnv/2020 vtune
# vtune-gui

== VDI nodes ==

# Connect to gra-vdi.alliancecan.ca with TigerVNC
# module load CcEnv StdEnv/2020 vtune
# vtune-gui

= MPI example =

First, load the latest VTune module.

 module load StdEnv/2020
 module load vtune

Then compile your MPI program as you usually would and run it inside a job or in an interactive session started by a salloc command using:

 srun aps your_mpi_program.x

After the program finishes, the profiling data will be stored in a directory called aps_result_YYYYMMDD where YYYYMMDD is the current date.

There is a lot of information you can extract from that data.  To get the basic summary report of your program's performance, run:

 aps-report  -D aps_result_YYYYMMDD

where you would replace YYYYMMDD to match the actual directory that has been created.  This command creates an HTML file, which can be copied to your own computer and viewed in a browser.  The report will clearly identify performance issues that are affecting your code.