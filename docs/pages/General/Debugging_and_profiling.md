---
title: "Debugging and profiling/en"
url: "https://docs.alliancecan.ca/wiki/Debugging_and_profiling/en"
category: "General"
last_modified: "2023-06-05T17:40:21Z"
page_id: 11634
display_title: "Debugging and profiling"
---

`<languages />`{=html} An important step in the software development process, particularly for compiled languages like Fortran and C/C++, concerns the use of a program called a debugger to detect and identify the origin of runtime errors (e.g. memory leaks, floating point exceptions and so forth) so that they can be eliminated. Once the program\'s correctness is assured, a further step is profiling the software. This involves the use of another software tool, a profiler, determine what percentage of the total execution time each section of the source code is responsible for when run with a representative test case. A profiler can give information like how many times a particular function is called, which other functions are calling it and how many milli-seconds of time each invocation of this function costs on average.

# Debugging and profiling tools {#debugging_and_profiling_tools}

Our national clusters offer a variety of debugging and profiling tools, both command line and those with a graphical user interface, whose use requires an X11 connection. Note that debugging sessions should be conducted using an [ interactive job](https://docs.alliancecan.ca/Running_jobs#Interactive_jobs " interactive job"){.wikilink} and not run on a login node.

## GNU debugger (gdb) {#gnu_debugger_gdb}

Please see the [ GDB page](https://docs.alliancecan.ca/GDB " GDB page"){.wikilink}.

## PGI debugger (pgdb) {#pgi_debugger_pgdb}

Please see the [Pgdbg page](https://docs.alliancecan.ca/wiki/Pgdbg).

## ARM debugger (ddt) {#arm_debugger_ddt}

Please see the [ ARM software page](https://docs.alliancecan.ca/ARM_software " ARM software page"){.wikilink}.

## GNU profiler (gprof) {#gnu_profiler_gprof}

Please see the [ Gprof page](https://docs.alliancecan.ca/Gprof " Gprof page"){.wikilink}.

## Scalasca profiler (scalasca, scorep, cube) {#scalasca_profiler_scalasca_scorep_cube}

Scalasca is an open source, GUI-driven parallel profiling tool set. It is currently available for `<b>`{=html}gcc 9.3.0`</b>`{=html} and `<b>`{=html}OpenMPI 4.0.3`</b>`{=html}, with AVX2 or AVX512 architecture. Its environment can be loaded with:

`module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 scalasca`

The current version is `<b>`{=html}2.5`</b>`{=html}. More information can be found in the 2.x user guide, which contains workflow examples [here](https://apps.fz-juelich.de/scalasca/releases/scalasca/2.5/docs/manual/).

## PGI profiler (pgprof) {#pgi_profiler_pgprof}

Please see the [ Pgprof page](https://docs.alliancecan.ca/PGPROF " Pgprof page"){.wikilink}.

## Nvidia command-line profiler (nvprof) {#nvidia_command_line_profiler_nvprof}

Please see the [ nvprof page](https://docs.alliancecan.ca/Nvprof " nvprof page"){.wikilink}.

## Valgrind

Please see the [ Valgrind page](https://docs.alliancecan.ca/Valgrind " Valgrind page"){.wikilink}.

# External references {#external_references}

- [Introduction to (Parallel) Performance](https://docs.scinet.utoronto.ca/index.php/Introduction_To_Performance) from SciNet
- [Code profiling on Graham](https://www.youtube.com/watch?v=YsF5KMr9uEQ), video, 54 minutes.
