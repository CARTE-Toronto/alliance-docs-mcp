---
title: "Biomolecular simulation/en"
url: "https://docs.alliancecan.ca/wiki/Biomolecular_simulation/en"
category: "General"
last_modified: "2025-02-10T19:07:41Z"
page_id: 5661
display_title: "Biomolecular simulation"
---

== General ==

Biomolecular simulationRon O. Dror, Robert M. Dirks, J.P. Grossman, Huafeng Xu, and David E. Shaw. "Biomolecular Simulation: A Computational Microscope for Molecular Biology." Annual Review of Biophysics,  41:429-452, 2012. https://doi.org/10.1146/annurev-biophys-042910-155245 is the application of molecular dynamics simulations to biochemical research questions. Processes that can be modeled include, but are not limited to, protein folding, drug binding, membrane transport, and the conformational changes critical to protein function.

While biomolecular simulation could be considered a sub-field of computational chemistry, it is sufficiently specialized that we have a Biomolecular Simulations National Team that supports this area. There is nevertheless some overlap of software tools between the two fields. See Computational chemistry for an annotated list of available software packages in that area.

== Software Packages ==

The following software packages are available on our HPC resources:

* AMBER
* GROMACS
* NAMD
* DL_POLY
* HOOMD-blue
* LAMMPS
* OpenKIM, the Knowledgebase of Interatomic Models
* OpenMM
* PLUMED, a library for code development related to the calculation of free energy in molecular dynamics simulations. See also GROMACS.
* Rosetta
* DSSP
* VMD

=== Python Packages (Python Wheels) ===

Our Wheelhouse contains a number of Python Wheels that can be installed within a virtual Python environment and are useful in the domain of biomolecular simulation/molecular dynamics.

This list contains a selection of useful wheels, but is not to be considered complete:

* ACPYPE: AnteChamber PYthon Parser interfacE is a tool to generate topologies for chemical compounds.
* MDAnalysis is an object-oriented Python library to analyze trajectories from molecular dynamics (MD) simulations in many popular formats.
* MDTraj can also read, write and analyze MD trajectories with only a few lines of Python code with wide MD format support.
* Biopython is a set of freely available tools for biological computation.
* foyer is a package for atom-typing as well as applying and disseminating force fields.
* mBuild is a hierarchical, component based molecule builder.
* mdsynthesis is a persistence engine for molecular dynamics data.
* nglview: NGL Viewer is a collection of tools for web-based molecular graphics.
* ParmEd is a general tool for aiding in investigations of biomolecular systems using popular molecular simulation packages.
* PyRETIS is a Python library for rare event molecular simulations with emphasis on methods based on transition interface sampling and replica exchange transition interface sampling.

Please check the list of available wheels and use the avail_wheels command on our clusters
to see what is available.

If you require additional Python packages or newer versions, please contact Support.

== Workshops and Training Material ==

The Molecular Modelling and Simulation National Team is offering Molecular Dynamics workshops.  Future workshops will be announced in our Newsletters.

The workshop material is also available for self-study:

# Practical considerations for Molecular Dynamics
#       Visualizing Structures with VMD
#         Running Molecular Dynamics with Amber on our clusters
#         Analyzing Molecular Dynamics Data with PYTRAJ

== Performance and benchmarking ==

A team at ACENET has created a Molecular Dynamics Performance Guide for Alliance clusters.
It can help you determine optimal conditions for Amber, GROMACS, NAMD, and OpenMM jobs.

== References ==