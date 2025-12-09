---
title: "OpenACC"
url: "https://docs.alliancecan.ca/wiki/OpenACC"
category: "General"
last_modified: "2020-11-30T17:54:41Z"
page_id: 15290
display_title: "OpenACC"
---

OpenACC makes it relatively easy to offload vectorized code to accelerators such as GPUs, for example. Unlike CUDA and OpenCL where kernels need to be coded explicitly, OpenACC minimizes the amount of modifications to do on a serial or OpenMP code. The compiler converts the OpenACC code into a binary executable that can make use of accelerators. The performance of OpenACC codes can be similar to the one of a CUDA code, except that OpenACC requires less code development.

= OpenACC directives =
Similar to OpenMP, OpenACC can convert a for loop into parallel code that would run on an accelerator. This can be achieved with compiler directives #pragma acc ... before structured blocks of code like, for example, a for loop. All supported pragma directives are described in the OpenACC specification.

= Code examples =
OpenACC can be used in Fortran, C and C++, which we illustrate here using a simple program that computes a decimal approximation to π based on a definite integral which is equal to arctan(1), i.e. π/4.

= Compilers =

== PGI ==
* Module pgi, any version from 13.10
** Newer versions support newest GPU capabilities.

Compilation example:
 # TODO

== GCC ==
* Module gcc, any version from 9.3.0
** Newer versions support newest GPU capabilities.

Compilation example:

 gcc -fopenacc -march=native -O3 pi.c -o pi

= Tutorial =
See our OpenACC_Tutorial.

= References =
* OpenACC official documentation - Specification 3.1 (PDF)
* NVIDIA OpenACC API - Quick Reference Guide (PDF)