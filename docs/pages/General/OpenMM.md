---
title: "OpenMM/en"
url: "https://docs.alliancecan.ca/wiki/OpenMM/en"
category: "General"
last_modified: "2024-10-16T16:58:41Z"
page_id: 19330
display_title: "OpenMM"
---

=Introduction=
OpenMMOpenMM home page: https://openmm.org/ is a toolkit for molecular simulation. It can be used either as a standalone application for running simulations or as a library you call from your own code. It provides a combination of extreme flexibility (through custom forces and integrators), openness, and high performance (especially on recent GPUs) that make it unique among MD simulation packages.

= Running a simulation with AMBER topology and restart files =

== Preparing the Python virtual environment ==

This example is for the openmm/7.7.0 module.

1. Create and activate the Python virtual environment.

2. Install ParmEd and netCDF4 Python modules.
3.4.3 netCDF4
}}

== Job submission ==
Below is a job script for a simulation using one GPU.

Here openmm_input.py is a Python script loading Amber files, creating the OpenMM simulation system, setting up the integration, and running dynamics. An example is available here.

= Performance and benchmarking =

A team at ACENET has created a Molecular Dynamics Performance Guide for Alliance clusters.
It can help you determine optimal conditions for AMBER, GROMACS, NAMD, and OpenMM jobs. The present section focuses on OpenMM performance.

OpenMM on the CUDA platform requires only one CPU per GPU because it does not use CPUs for calculations. While OpenMM can use several GPUs in one node, the most efficient way to run simulations is to use a single GPU. As you can see from  Narval benchmarks and  Cedar benchmarks, on nodes with NvLink (where GPUs are connected directly), OpenMM runs slightly faster on multiple GPUs. Without NvLink there is a very little speedup of simulations on P100 GPUs (Cedar benchmarks).