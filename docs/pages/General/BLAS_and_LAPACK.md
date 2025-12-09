---
title: "BLAS and LAPACK/en"
url: "https://docs.alliancecan.ca/wiki/BLAS_and_LAPACK/en"
category: "General"
last_modified: "2025-01-06T16:20:02Z"
page_id: 18718
display_title: "BLAS and LAPACK"
---

`<languages />`{=html} [BLAS (Basic Linear Algebra Subprogram](http://www.netlib.org/blas/) and [LAPACK (Linear Algebra PACK)](http://www.netlib.org/lapack/) are two of the most commonly used libraries in advanced research computing. They are used for vector and matrix operations that are commonly found in a plethora of algorithms. More importantly, they are more than libraries, as they define a standard programming interface. A programming interface is a set of function definitions that can be called to accomplish specific computation, for example the dot product of two vectors of double precision numbers, or the matrix product of two hermitian matrices of complex numbers.

Beside the reference implementation done by Netlib, there exist a large number of implementations of these two standards. The performance of these implementations can vary widely depending on the hardware that is running them. For example, it is well established that the implementation provided by the [Intel Math Kernel Library (Intel MKL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) performs best in most situations on Intel processors. That implementation is however proprietary, and in some situations, it is preferred to use the open source implementation [OpenBLAS](https://github.com/xianyi/OpenBLAS). Another open source implementation, named [BLIS](https://github.com/flame/blis), performs better on AMD processors. Previously, you may have known [gotoblas](https://www.tacc.utexas.edu/research-development/tacc-software/gotoblas2) and [ATLAS BLAS](https://github.com/math-atlas/math-atlas), but those projects are no longer maintained.

Unfortunately, testing which implementation performs best for a given code and given hardware usually requires recompiling software. This is a problem when trying to create a portable software environment that works on multiple clusters. This can be fixed by using [FlexiBLAS](https://www.mpi-magdeburg.mpg.de/projects/flexiblas). This is an abstraction layer that allows one to swap which implementation of BLAS and LAPACK is used at runtime, rather than at compile time.

# Which implementation should I use ? {#which_implementation_should_i_use}

For the past few years, we have been recommending to use Intel MKL as a reference implementation. This recommendation was driven by the fact that we only had Intel processors in our clusters. This changed with the arrival of [Narval](https://docs.alliancecan.ca/Narval/en "Narval"){.wikilink}, which is built with AMD processors. We now recommend using FlexiBLAS when compiling code. Our FlexiBLAS module is configured such that Intel MKL will be used except when using AMD processors, in which case BLIS will be used. This arrangement will usually offer optimal performance.

# How do I compile against FlexiBLAS ? {#how_do_i_compile_against_flexiblas}

Unfortunately, FlexiBLAS is relatively new, and not all build systems will recognize it by default. This can generally be fixed by setting the linking options to use `-lflexiblas` for BLAS and for LAPACK. You will typically find these options in your Makefile, or be able to pass them as parameters to `configure` or `cmake`. Versions 3.19 and higher of CMake can find FlexiBLAS automatically; you must load one of the `cmake/3.20.1` or `cmake/3.21.4` modules to use such a version.

# How do I change which implementation of BLAS/LAPACK is used at run time ? {#how_do_i_change_which_implementation_of_blaslapack_is_used_at_run_time}

The main benefit of using FlexiBLAS is the ability to change the implementation backend at run time by setting the environment variable `FLEXIBLAS`. At the time of this writing, four implementations are available: `netlib`, `blis`, `imkl` and `openblas`, but the full list can be obtained by running the command

On [Narval](https://docs.alliancecan.ca/Narval/en "Narval"){.wikilink}, we have set `FLEXIBLAS=blis` to use BLIS by default, while on other clusters, `FLEXIBLAS` is left undefined, which defaults to using Intel MKL.

# Using Intel MKL directly {#using_intel_mkl_directly}

Although we recommend using FlexiBLAS, it is still possible to use Intel MKL directly. If you are using one of the Intel compilers (e.g. `ifort, icc, icpc`) then the solution is to replace `-lblas` and `-llapack` in your compiler and linker options with either:

- `-mkl=sequential`, which will not use internal threading, or
- `-mkl` which will use internal threading.

This will ensure that the MKL implementation of BLAS/LAPACK is used. See [here](https://software.intel.com/en-us/mkl-linux-developer-guide-using-the-mkl-compiler-option) for more on the significance of `sequential` and other options.

If you are using a non-Intel compiler, for example the GNU Compiler Collection, then you will need to explicitly list the necessary MKL libraries during the link phase. Intel provides a tool called the [MKL Link Advisor](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor) to help you find the correct compiler and linker options.

The same [MKL Link Advisor](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor) tool is also useful if you receive \"undefined reference\" errors while using Intel compilers and `-mkl`.
