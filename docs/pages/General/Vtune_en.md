---
title: "Vtune/en"
url: "https://docs.alliancecan.ca/wiki/Vtune/en"
category: "General"
last_modified: "2024-02-14T15:15:05Z"
page_id: 15889
display_title: "Vtune"
---

`<languages />`{=html}

\_\_FORCETOC\_\_

# Introduction

[VTune](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/vtune-profiler.html) is Intel\'s Performance Analysis tool for applications and systems. It is capable of [Analyzing both OpenMP and MPI](https://software.intel.com/content/www/us/en/develop/documentation/itac-vtune-mpi-openmp-tutorial-lin/top.html) based applications.

# Software module {#software_module}

To load the module on any Alliance cluster run:

# Tool renaming {#tool_renaming}

The content of this page is largely concerned with the legacy version named Intel® VTune™ Amplifier. Please note this tool has been renamed throughout Intel\'s documentation in latest versions (newer than the latest `vtune` module versions presently available on Alliance clusters) from Intel® VTune™ Amplifier to Intel® VTune™ Profiler. Likewise the application commands amplxe-cl and amplxe-gui have been renamed to vtune and vtune-gui for both the command line and graphical tools respectively. Further information can be found [here](https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top/launch.html).

# Analysis types {#analysis_types}

To collect analysis information run:

where `<analysis-type>`{=html} should be replaced by one of the available analysis, e.g. hotspots, and `<target_exe>`{=html} is the path to the executable you would like to analyze. It is recommended to compile your executable with the \"-g\" option and to use the same optimization level as normal so as to obtain accurate results. A listing of version specific argument options and several usage examples maybe displayed on the command line by running `vtune -help`, after loading the vtune module. Complete downloadable documentation for Parallel Studio XE (including VTune) for all recent versions can be found [here](https://software.intel.com/content/www/us/en/develop/articles/download-documentation-intel-parallel-studio-xe-current-previous.html). The latest version of the Intel VTune Profiler User Guide may be found [here](https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top.html).

# Create reports {#create_reports}

To create a report run this command:

where `<report-type>`{=html} is the type of the report to generate, e.g. hotspots. See also:

- [<https://software.intel.com/en-us/vtune-amplifier-help-generating-command-line-reports>](https://software.intel.com/en-us/vtune-amplifier-help-generating-command-line-reports)

# Matrix example {#matrix_example}

Analyze and generate a summary report for the Intel Matrix Sample Project run from the command line with 4 cores:

`salloc --time=1:00:00 --cpus-per-task=4 --ntasks=1 --mem=16G --account=def-yours`\
`module load StdEnv/2020 vtune`\
`cp -a $EBROOTVTUNE/vtune/$EBVERSIONVTUNE*/samples/en/C++/matrix . cd matrix/linux`\
`make icc`\
`vtune -collect hotspots ../matrix`\
`vtune -report summary `

The latest version of matrix_multiply (uses cmake to build) can be found [here](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/VTuneProfiler).

# Graphical mode {#graphical_mode}

The Intel Matrix Sample Project can also be run using Vtune in GUI mode as explored here [1](https://software.intel.com/content/www/us/en/develop/documentation/vtune-hotspots-tutorial-linux-c/top/run-hotspots-analysis.html). To run VTune over VNC follow the below directions depending on which system you wish to use. Running VTune graphically can be useful to generate command line configurations as discussed in [2](https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top/analyze-performance/control-data-collection/generating-command-line-configuration-from-gui.html).

## Cluster nodes {#cluster_nodes}

1.  Connect to a cluster compute or login node with [TigerVNC](https://docs.alliancecan.ca/wiki/VNC#Connect)
2.  `module load StdEnv/2020 vtune`
3.  `vtune-gui`\

## VDI nodes {#vdi_nodes}

1.  Connect to gra-vdi.alliancecan.ca with [TigerVNC](https://docs.alliancecan.ca/wiki/VNC#VDI_Nodes)
2.  `module load CcEnv StdEnv/2020 vtune`
3.  `vtune-gui`\

# MPI example {#mpi_example}

First, load the latest VTune module.

`module load StdEnv/2020`\
`module load vtune`

Then compile your MPI program as you usually would and run it inside a job or in an interactive session started by a salloc command using:

`srun aps your_mpi_program.x`

After the program finishes, the profiling data will be stored in a directory called aps_result_YYYYMMDD where YYYYMMDD is the current date.

There is a lot of information you can extract from that data. To get the basic summary report of your program\'s performance, run:

`aps-report  -D aps_result_YYYYMMDD`

where you would replace YYYYMMDD to match the actual directory that has been created. This command creates an HTML file, which can be copied to your own computer and viewed in a browser. The report will clearly identify performance issues that are affecting your code.

![Example report generated by aps for an MPI program](https://docs.alliancecan.ca/Vtune_mpi_profiling_report.png "Example report generated by aps for an MPI program"){width="800"}
