---
title: "PGPROF/en"
url: "https://docs.alliancecan.ca/wiki/PGPROF/en"
category: "General"
last_modified: "2019-09-23T13:08:24Z"
page_id: 9424
display_title: "PGPROF"
---

`<languages />`{=html}

PGPROF is a powerful and simple tool for analyzing the performance of parallel programs written with OpenMP, MPI, OpenACC, or CUDA. There are two profiling modes: Command-line mode and graphical mode.

# Quickstart guide {#quickstart_guide}

Using PGPROF usually consists of two steps:

1.  **Data collection**: Run the application with profiling enabled.
2.  **Analysis**: Visualize the data produced in the first step.

Both steps can be accomplished in either command-line mode or graphical mode.

## Environment modules {#environment_modules}

Before you start profiling with PGPROF, the appropriate [module](https://docs.alliancecan.ca/Utiliser_des_modules/en "module"){.wikilink} needs to be loaded.

PGPROF is part of the PGI compiler package, so run `module avail pgi` to see what versions are currently available with the compiler, MPI, and CUDA modules you have loaded. For a comprehensive list of PGI modules, run `module -r spider '.*pgi.*'`.\
As of December 2018, these were:

- pgi/13.10
- pgi/17.3

Use `module load pgi/version` to select a version; for example, to load the PGI compiler version 17.3, use

## Compiling your code {#compiling_your_code}

To get useful information from PGPROF, you first need to compile your code with one of the PGI compilers (`pgcc` for C, `pgc++` for C++ , `pgfortran` for Fortran). A source in Fortran may need to be compiled with the `-g` flag.

## Command-line mode {#command_line_mode}

**Data collection**: Use PGPROF to run the application and save the performance data in a file. In this example, the application is `a.out` and we choose to save the data in `a.prof`.

The data file can be analyzed in graphical mode with the *File \| Import* command (see below) or in command-line mode as follows.\
\
**Analysis**: To visualize the performance data in command-line mode:

The results are usually divided into several categories, for example:

- GPU kernel execution profile
- CUDA API execution profile
- OpenACC execution profile
- CPU execution profile

### Options

- The output can be cropped to show one of the categories. For example, the option `--cpu-profiling` will show only the CPU results.

<!-- -->

- The option `--cpu-profiling-mode top-down` will make the PGPROF show the main subroutine at the top and the rest of functions it called below:

<!-- -->

- To find out what part of your application takes the longest time to run you can use the option `--cpu-profiling-mode bottom-up` which orients the call tree to show each function followed by functions that called it and working backwards to the main function.

## Graphical mode {#graphical_mode}

![Starting a new PGPROF session (click for a larger image)\|left ](https://docs.alliancecan.ca/pgprof-start-session.png "Starting a new PGPROF session (click for a larger image)|left "){width="300"} In graphical mode, both data collection and analysis can be accomplished in the same session most of the time. However, it is also possible to do the analysis from the pre-saved performance data file (e.g. collected in the command-line mode). There are several steps that need to be done to collect and visualize performance data in this mode.\
\
**Data collection**

- Launch the PGI profiler.
  - Since the Pgrof\'s GUI is based on Java, it should be executed on the compute node in the interactive session rather than on the login node, as the latter does not have enough memory (see [Java](https://docs.alliancecan.ca/Java#Pitfalls "Java"){.wikilink} for more details). An interactive session can be started with `salloc --x11 ...` to enable X11 forwarding (see [Interactive jobs](https://docs.alliancecan.ca/Running_jobs#Interactive_jobs "Interactive jobs"){.wikilink} for more details).
- In order to start a new session, open the *File* menu and click on *New Session*.
- Select the executable file you want to profile and then add any arguments appropriate for your profiling.
- Click *Next*, then *Finish*.

**Analysis**\
In the *CPU Details* tab, click on the *Show the top-down (callers first) call tree view* button.

![Visualizing performance data (click for a larger image)\|left ](https://docs.alliancecan.ca/pgprof2.png "Visualizing performance data (click for a larger image)|left "){width="300"} The visualization window is comprised of four panes:\
- The pane on the upper right shows the timeline with all the events ordered by the time at which they were executed.\
- **GPU Details**: shows performance details for the GPU kernels.\
- **CPU Details**: shows performance details for the CPU functions.\
- **Properties**: shows all the details for a selected function in the timeline window.\

# References

PGPROF is a product of [PGI](https://www.pgroup.com/), which is a subsidiary of [NVIDIA Corporation.](https://www.nvidia.com)

- [Quick Start Guide](https://www.pgroup.com/resources/pgprof-quickstart.htm)
- [User\'s Guide](https://www.pgroup.com/doc/pgi17profug.pdf)
