---
title: "OpenACC Tutorial - Adding directives/en"
url: "https://docs.alliancecan.ca/wiki/OpenACC_Tutorial_-_Adding_directives/en"
category: "User Guide"
last_modified: "2023-06-08T19:37:05Z"
page_id: 1447
display_title: "OpenACC Tutorial - Adding directives"
---

== Offloading to a GPU ==
The first thing to realize when trying to port a code to a GPU is that both the CPU located in the host and the GPU do not share the same memory:
* The host memory is generally larger, but slower than the GPU memory;
* A GPU does not have direct access to the host memory;
* To use a GPU, data must therefore be transferred from the main program to the GPU through the PCI bus, which has a much lower bandwidth than either memories.
This means that managing data transfers between the host and the GPU will be of paramount importance. Transferring the data and the code onto the device is called offloading.

== OpenACC directives ==
OpenACC directives are much like OpenMP directives.
They take the form of pragma statements in C/C++, and comments in Fortran.
There are several advantages to using directives:
* First, since it involves very minor modifications to the code, changes can be done incrementally, one pragma at a time. This is especially useful for debugging purpose, since making a single change at a time allows one to quickly identify which change created a bug.
* Second, OpenACC support can be disabled at compile time. When OpenACC support is disabled, the pragma are considered comments, and ignored by the compiler. This means that a single source code can be used to compile both an accelerated version and a normal version.
* Third, since all of the offloading work is done by the compiler, the same code can be compiled for various accelerator types: GPUs or SIMD instructions on CPUs. It also means that a new generation of devices only requires one to update the compiler, not to change the code.

In the following example, we take a code comprised of two loops.
The first one initializes two vectors, and the second performs a SAXPY, a basic vector addition operation.

C/C++	FORTRAN

#pragma acc kernels
{
  for (int i=0; i prevents parallelization
              Loop carried backward dependence of ycoefs-> prevents vectorization
              Complex loop carried dependence of Acoefs->,xcoefs-> prevents parallelization
              Generating NVIDIA GPU code
              31, #pragma acc loop seq
              35, #pragma acc loop vector(128) /* threadIdx.x */
                  Generating implicit reduction(+:sum)
          35, Loop is parallelizable
}}

As we can see in the compiler output, the compiler could not parallelize the outer loop on line 31.
We will see in the following sections how to deal with those dependencies.

== Fixing false loop dependencies ==
Sometimes, the compiler believes that loops cannot be parallelized despite being obvious to the programmer. One common case, in C and C++, is what is called pointer aliasing. Contrary to Fortran arrays, C and C++ do not formally have arrays. They have what is called pointers. Two pointers are said to be aliased if they point to the same memory. If the compiler does not know that pointers are not aliased, it must assume that they are. Going back to the previous example, it becomes obvious why the compiler could not parallelize the loop. If we assume that each pointer is the same, then there is an obvious dependence between loop iterations.

=== restrict keyword ===
One way to tell the compiler that pointers are not going to be aliased, is by using a special keyword. In C, the keyword restrict was introduced in C99 for this purpose.  In C++, there is no standard way yet, but each compiler typically has its own keyword. Either __restrict or __restrict__ can be used depending on the compiler. For Portland Group and NVidia compilers, the keyword is __restrict. For an explanation as to why there is no standard way to do this in C++, you can read this paper. This concept is important not only for OpenACC, but for any C/C++ programming, since many more optimizations can be done by compilers when pointers are guaranteed not to be aliased. Note that the keyword goes after the pointer, since it refers to the pointer, and not to the type. In other words, you would declare float * __restrict A; rather than float __restrict * A;.

=== Loop directive with independent clause ===
Another way to tell the compiler that loops iterations are independent is to specify it explicitly by using a different directive: loop, with the clause independent. This is a prescriptive directive. Like any prescriptive directive, this tells the compiler what to do, and overrides any compiler analysis. The initial example above would become:

#pragma acc kernels
{
#pragma acc loop independent
for (int i=0; i New Session, or click on the corresponding button in the toolbar.
# Click on the Browse button at the right of the File path editor.
## Change directory if needed.
## Select an executable built from codes written with OpenACC and CUDA C/C++ instructions.
# Below the Arguments editor, select the profiling option Profile current process only.
# Click Next > to review additional profiling options.
# Click Finish to start profiling the executable.

This can be done with the following steps:
# Start nvvp with the command nvvp &   (the & sign is to start it in the background)
# Go in File -> New Session
# In the "File:" field, search for the executable (named challenge in our example).
# Click "Next" until you can click "Finish".

This will run the program and generate a timeline of the execution. The resulting timeline is illustrated on the image on the right side. As we can see, almost all of the run time is being spent transferring data between the host and the device. This is very often the case when one ports a code from CPU to GPU. We will look at how to optimize this in the next part of the tutorial.

== The parallel loop directive ==
With the kernels directive, we let the compiler do all of the analysis. This is the descriptive approach to porting a code. OpenACC supports a prescriptive approach through a different directive, called the parallel directive. This can be combined with the loop directive, to form the parallel loop directive. An example would be the following code:

#pragma acc parallel loop
for (int i=0; i and -Minfoaccel to your compiler flags.
}}

<- Previous unit: Profiling | ^- Back to the lesson plan | Onward to the next unit: Data movement ->