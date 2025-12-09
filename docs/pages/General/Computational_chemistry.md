---
title: "Computational chemistry/en"
url: "https://docs.alliancecan.ca/wiki/Computational_chemistry/en"
category: "General"
last_modified: "2025-01-27T16:17:17Z"
page_id: 5636
display_title: "Computational chemistry"
---

Computational chemistry is a branch of chemistry that incorporates the results of theoretical chemistry into computer programs to calculate the structures and properties of molecules and solids.

Most computer programs in the field offer a large number of methods, which can be broadly grouped in terms of the trade-off between accuracy, applicability, and cost.
*ab initio methods, based entirely on first principles, tend to be broadly applicable but very costly in terms of CPU time; they are therefore mostly applied to systems with a small number of particules.
*Semi-empirical methods give accurate results for a narrower range of cases, but are also typically much faster than ab initio methods.
*Density functional methods may be thought of as a compromise in cost between ab initio and semi-empirical methods. The cost-accuracy trade-off is very good and density functional methods have therefore become very widely used in recent years.
*Molecular mechanics methods, based on classical mechanics instead of quantum mechanics, are faster but more narrowly applicable. They use a force field that can be optimized using ab initio and/or experimental data to reproduce the properties of the materials. Because of the low cost, molecular mechanics methods are frequently used for molecular dynamics calculations and can be applied to systems of thousands or even millions of particles.

Molecular dynamics calculations are extremely useful in the study of biological systems. Please see the Biomolecular simulation page for a list of the resources relevant to this area of research, but bear in mind that the distinction is artificial and many tools are applicable to both biological and non-biological systems. They can be used to simulate glasses, metals, liquids, supercooled liquids, granular materials, complex materials, etc.

=== Notes on installed software ===

==== Applications ====

* ABINIT
* ADF/AMS
* AMBER
* CP2K
* CPMD
* Dalton
* deMon
* DL_POLY
* GAMESS-US
* Gaussian
* GPAW
* GROMACS
* HOOMD-blue
* LAMMPS
* MRCC
* NAMD
* NBO is included in several of our Gaussian modules.
* NWChem
* OpenKIM
* OpenMM
* ORCA
* PLUMED
* PSI4
* Quantum ESPRESSO
* Rosetta
* SIESTA
* VASP
* XTB (Extended Tight Binding)

An automatically generated list of all the versions installed on Compute Canada systems can be found on Available software.

====Visualization tools====
*Molden, a visualization tool for use in conjunction with GAMESS, Gaussian and other applications.
*VMD, an open-source molecular visualization program for displaying, animating, and analyzing large biomolecular systems in 3D.
*VisIt, a general-purpose 3D visualization tool (a gallery presents examples from chemistry).
See Visualization for more about producing visualizations on Compute Canada clusters.

==== Other tools ====
* CheMPS2, a "library which contains a spin-adapted implementation of the density matrix renormalization group (DMRG) for ab initio quantum chemistry."
* Libxc, a library used in density-functional models.
* Open3DQSAR, a "tool aimed at pharmacophore exploration by high-throughput chemometric analysis of molecular interaction fields."
* Open Babel, a set of tools to enable one "to search, convert, analyze, or store data from molecular modeling, chemistry, solid-state materials, biochemistry, or related areas."
* PCMSolver, a tool for code development related to the Polarizable Continuum Model. Some applications listed above offer built-in capabilities related to the PCM.
* RDKit, a collection of cheminformatics and machine-learning software written in C++ and Python.
* Spglib, a library for development relating to the symmetry of crystals.