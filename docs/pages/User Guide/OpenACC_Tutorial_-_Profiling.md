---
title: "OpenACC Tutorial - Profiling/en"
url: "https://docs.alliancecan.ca/wiki/OpenACC_Tutorial_-_Profiling/en"
category: "User Guide"
last_modified: "2022-12-20T22:27:07Z"
page_id: 1448
display_title: "OpenACC Tutorial - Profiling"
---

== Code profiling ==
Why would one need to profile code? Because it's the only way to understand:
* Where time is being spent (hotspots)
* How the code is performing
* Where to focus your development time

What is so important about hotspots in the code?
The Amdahl's law says that
"Parallelizing the most time-consuming routines (i.e. the hotspots) will have the most impact".

== Build the Sample Code ==
For the following example, we use a code from this Git repository.
You are invited to download and extract the package, and go to the cpp or the f90 directory.
The object of this example is to compile and link the code, obtain an executable, and then profile its source code with a profiler.

Once the executable cg.x is created, we are going to profile its source code:
the profiler will measure function calls by executing and monitoring this program.
Important: this executable uses about 3GB of memory and one CPU core at near 100%.
Therefore, a proper test environment should have at least 4GB of available memory and at least two (2) CPU cores.

=== NVIDIA nvprof Command Line Profiler ===
NVIDIA usually provides nvprof with its HPC SDK,
but the proper version to use on our clusters is included with a CUDA module:

To profile a pure CPU executable, we need to add the arguments --cpu-profiling on to the command line:
 main
  7.94%  8.62146s  waxpby(double, vector const &, double, vector const &, vector const &)
  7.94%  8.62146s   main
  5.86%  6.36584s  dot(vector const &, vector const &)
  5.86%  6.36584s   main
  2.47%  2.67666s  allocate_3d_poisson_matrix(matrix&, int)
  2.47%  2.67666s   main
  0.13%  140.35ms  initialize_vector(vector&, double)
  0.13%  140.35ms   main
...
======== Data collected at 100Hz frequency
}}
From the above output, the matvec() function is responsible for 83.5% of the execution time, and this function call can be found in the main() function.

== Compiler Feedback ==
Before working on the routine, we need to understand what the compiler is actually doing by asking ourselves the following questions:
* What optimizations were applied automatically by the compiler?
* What prevented further optimizations?
* Can very minor modifications of the code affect performance?

The NVIDIA compiler offers a -Minfo flag with the following options:
* all - Print almost all types of compilation information, including:
** accel - Print compiler operations related to the accelerator
** inline - Print information about functions extracted and inlined
** loop,mp,par,stdpar,vect - Print various information about loop optimization and vectorization
* intensity - Print compute intensity information about loops
* (none) - If -Minfo is used without any option, it is the same as with the all option, but without the inline information

=== How to Enable Compiler Feedback ===
* Edit the Makefile:
  CXX=nvc++
  CXXFLAGS=-fast -Minfo=all,intensity
  LDFLAGS=${CXXFLAGS}

* Rebuild

=== Interpretation of the Compiler Feedback ===
The Computational Intensity of a loop is a measure of how much work is being done compared to memory operations.
Basically:

\mbox{Computational Intensity} = \frac{\mbox{Compute Operations}}{\mbox{Memory Operations}}

In the compiler feedback, an Intensity \ge 1.0 suggests that the loop might run well on a GPU.

== Understanding the code  ==
Let's look closely at the main loop in the
matvec() function implemented in matrix_functions.h:

  for(int i=0;i