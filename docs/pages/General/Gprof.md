---
title: "Gprof/en"
url: "https://docs.alliancecan.ca/wiki/Gprof/en"
category: "General"
last_modified: "2020-10-13T17:50:03Z"
page_id: 14784
display_title: "Gprof"
---

=GNU Profiler (gprof) =
== What is gprof? ==
gprof is a profiling software which collects information and compiles statistics on your code. Generally, it searches for functions and subroutines in your program and inserts timing instructions for each one. Executing such a modified program creates a raw data file which can be interpreted by gprof and turned into profiling statistics.

gprof comes with the GNU compiler suite and is available with the gcc module on our clusters.

== Preparing your application ==
=== Loading the GNU compiler ===
Load the appropriate GNU compiler. For example, for GCC:

=== Compiling your code ===
To get useful information from gprof, you first need to compile your code with debugging information enabled. With the GNU compilers, you do so by adding the -pg option to the compilation command. This option tells the compiler to generate extra code to write profile information suitable for the analysis. Without this option, no call-graph data will be gathered and you may get the following error:

gprof: gmon.out file is missing call-graph data

=== Executing your code ===
Once your code has been compiled with the proper options, you then execute it:

You should run your code the same way as you would without gprof profiling; the execution line does not change.
Once the binary has been executed and finished without any errors, a new file gmon.out is created in the current working directory. Note that if your code changes the current directory, gmon.out will be created in the new working directory, insofar as your program has sufficient permissions to do so.

=== Getting the profiling data ===
In this step the gprof tool is executed with the binary name and the above mentioned gmon.out as argument; the analysis file is created with all the desired profiling information.