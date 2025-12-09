---
title: "OpenACC Tutorial/en"
url: "https://docs.alliancecan.ca/wiki/OpenACC_Tutorial/en"
category: "User Guide"
last_modified: "2024-10-16T16:56:41Z"
page_id: 1390
display_title: "OpenACC Tutorial"
---

This tutorial is strongly inspired from the OpenACC Bootcamp session presented at GPU Technology Conference 2016.

OpenACC is an application programming interface (API) for porting code onto accelerators such as GPU and coprocessors. It has been developed by Cray, CAPS, NVidia and PGI. Like in OpenMP, the programmer annotates C, C++ or Fortran code to identify portions that should be parallelized by the compiler.

A self-paced course on this topic is available from SHARCNET: Introduction to GPU Programming.

== Lesson plan ==
* Introduction
* Gathering a profile and getting compiler information
* Expressing parallelism with OpenACC directives
* Expressing data movement
* Optimizing loops

== External references ==
Here are some useful external references:
* OpenACC Programming and Best Practices Guide (PDF)
* OpenACC API 2.7 Reference Guide (PDF)
* Getting Started with OpenACC
* PGI Compiler
* PG Profiler
* NVIDIA Visual Profiler