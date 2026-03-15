---
title: "OpenMM"
url: "https://docs.alliancecan.ca/wiki/OpenMM"
category: "General"
last_modified: "2026-03-09T21:46:17Z"
page_id: 19215
display_title: "OpenMM"
---

=Introduction=
OpenMMOpenMM home page: https://openmm.org/ is an open-source molecular dynamics toolkit designed for flexibility and programmability. It is used via Python, offering both application-level classes for running simulations and a lower-level API that allows users to integrate OpenMM directly into their own code for custom workflows. OpenMM can natively read and simulate systems prepared with AMBER, GROMACS, and CHARMM, enabling seamless reuse of existing biomolecular setups. Its plugin architecture supports integration with machine-learning potentials, including TorchMD-Net, MACE, TorchANI, AIMNet2, and DeepMD for general-purpose or hybrid ML/MM simulations.

== Strengths ==
* Flexible Python interface with both high-level classes and low-level API access for custom workflows.
* High-level plugin framework for ML-driven potentials and hybrid simulations.
* Efficient execution on CPUs and GPUs, suitable for HPC platforms.
* Native support for major biomolecular formats (AMBER, GROMACS, CHARMM).
* Open-source with an active ecosystem of plugins for ML and advanced force fields.

== Weak points ==
* Slower than highly optimized classical MM engines (GROMACS, AMBER) for large-scale production runs.
* Flexibility can add complexity for hybrid ML/MM simulations.
* Specialized trajectory analysis may require external tools.

=  Environment modules =

$ module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.6 openmm/8.4.0 ambertools/25.0

Note: The ambertools module is optional and required only if you plan to simulate AMBER-prepared systems.

Optionally, create a Python virtual environment if you want to install extra packages (e.g., ML potentials).

= Preparing Input Files =

OpenMM can directly read Amber topology and coordinate/restart files if simulating AMBER systems.

Ensure your system is equilibrated and minimized in Amber or another package before transferring files to HPC.

For GROMACS or CHARMM systems, OpenMM can read their respective formats without AmberTools.

= Job submission =
Below is a job script for a simulation using one GPU.

= Python simulation script =
The example Python script below loads Amber parameter and restart files, builds the OpenMM simulation system, sets up the integrator, and runs the dynamics.

= Performance and benchmarking =

A team at ACENET has created a Molecular Dynamics Performance Guide for Alliance clusters.
It can help you determine optimal conditions for AMBER, GROMACS, NAMD, and OpenMM jobs. The present section focuses on OpenMM performance.

OpenMM on the CUDA platform requires only one CPU per GPU because it does not use CPUs for calculations. While OpenMM can use several GPUs in one node, the most efficient way to run simulations is to use a single GPU. As you can see from  Narval benchmarks and  Cedar benchmarks, on nodes with NvLink (where GPUs are connected directly), OpenMM runs slightly faster on multiple GPUs. Without NvLink there is a very little speedup of simulations on P100 GPUs (Cedar benchmarks).