---
title: "Computational chemistry/en"
url: "https://docs.alliancecan.ca/wiki/Computational_chemistry/en"
category: "General"
last_modified: "2025-01-27T16:17:17Z"
page_id: 5636
display_title: "Computational chemistry"
---

`<languages />`{=html}

[Computational chemistry](https://en.wikipedia.org/wiki/Computational_chemistry) is a branch of chemistry that incorporates the results of theoretical chemistry into computer programs to calculate the structures and properties of molecules and solids.

Most computer programs in the field offer a large number of methods, which can be broadly grouped in terms of the trade-off between accuracy, applicability, and cost.

- [*ab initio*](https://en.wikipedia.org/wiki/Ab_initio_quantum_chemistry_methods) methods, based entirely on first principles, tend to be broadly applicable but very costly in terms of CPU time; they are therefore mostly applied to systems with a small number of particules.
- [Semi-empirical](https://en.wikipedia.org/wiki/Semi-empirical_quantum_chemistry_method) methods give accurate results for a narrower range of cases, but are also typically much faster than *ab initio* methods.
- [Density functional](https://en.wikipedia.org/wiki/Density_functional_theory) methods may be thought of as a compromise in cost between *ab initio* and semi-empirical methods. The cost-accuracy trade-off is very good and density functional methods have therefore become very widely used in recent years.
- [Molecular mechanics](https://en.wikipedia.org/wiki/Molecular_mechanics) methods, based on classical mechanics instead of quantum mechanics, are faster but more narrowly applicable. They use a force field that can be optimized using *ab initio* and/or experimental data to reproduce the properties of the materials. Because of the low cost, molecular mechanics methods are frequently used for molecular dynamics calculations and can be applied to systems of thousands or even millions of particles.

Molecular dynamics calculations are extremely useful in the study of biological systems. Please see the [Biomolecular simulation](https://docs.alliancecan.ca/Biomolecular_simulation "Biomolecular simulation"){.wikilink} page for a list of the resources relevant to this area of research, but bear in mind that the distinction is artificial and many tools are applicable to both biological and non-biological systems. They can be used to simulate glasses, metals, liquids, supercooled liquids, granular materials, complex materials, etc.

### Notes on installed software {#notes_on_installed_software}

#### Applications

- [ABINIT](https://docs.alliancecan.ca/ABINIT "ABINIT"){.wikilink}
- [ADF](https://docs.alliancecan.ca/ADF "ADF"){.wikilink}/[AMS](https://docs.alliancecan.ca/AMS "AMS"){.wikilink}
- [AMBER](https://docs.alliancecan.ca/AMBER "AMBER"){.wikilink}
- [CP2K](https://docs.alliancecan.ca/CP2K "CP2K"){.wikilink}
- [CPMD](https://docs.alliancecan.ca/CPMD "CPMD"){.wikilink}
- [Dalton](https://docs.alliancecan.ca/Dalton "Dalton"){.wikilink}
- [deMon](http://www.demon-software.com/public_html/program.html)
- [DL_POLY](https://docs.alliancecan.ca/DL_POLY "DL_POLY"){.wikilink}
- [GAMESS-US](https://docs.alliancecan.ca/GAMESS-US "GAMESS-US"){.wikilink}
- [Gaussian](https://docs.alliancecan.ca/Gaussian "Gaussian"){.wikilink}
- [GPAW](https://docs.alliancecan.ca/GPAW "GPAW"){.wikilink}
- [GROMACS](https://docs.alliancecan.ca/GROMACS "GROMACS"){.wikilink}
- [HOOMD-blue](http://glotzerlab.engin.umich.edu/hoomd-blue/)
- [LAMMPS](https://docs.alliancecan.ca/LAMMPS "LAMMPS"){.wikilink}
- [MRCC](https://docs.alliancecan.ca/MRCC "MRCC"){.wikilink}
- [NAMD](https://docs.alliancecan.ca/NAMD "NAMD"){.wikilink}
- [NBO](https://nbo7.chem.wisc.edu/) is included in several of our [Gaussian](https://docs.alliancecan.ca/Gaussian#Notes "Gaussian"){.wikilink} modules.
- [NWChem](http://www.nwchem-sw.org)
- [OpenKIM](https://openkim.org/)
- [OpenMM](https://simtk.org/home/openmm)
- [ORCA](https://docs.alliancecan.ca/ORCA "ORCA"){.wikilink}
- [PLUMED](http://www.plumed-code.org)
- [PSI4](http://www.psicode.org/)
- [Quantum ESPRESSO](https://docs.alliancecan.ca/Quantum_ESPRESSO "Quantum ESPRESSO"){.wikilink}
- [Rosetta](https://www.rosettacommons.org)
- [SIESTA](http://departments.icmab.es/leem/siesta)
- [VASP](https://docs.alliancecan.ca/VASP "VASP"){.wikilink}
- [XTB (Extended Tight Binding)](https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/xtb)

An automatically generated list of all the versions installed on Compute Canada systems can be found on [Available software](https://docs.alliancecan.ca/Available_software "Available software"){.wikilink}.

#### Visualization tools {#visualization_tools}

- [Molden](https://www.theochem.ru.nl/molden/), a visualization tool for use in conjunction with GAMESS, Gaussian and other applications.
- [VMD](https://docs.alliancecan.ca/Visualization#VMD "VMD"){.wikilink}, an open-source molecular visualization program for displaying, animating, and analyzing large biomolecular systems in 3D.
- [VisIt](https://docs.alliancecan.ca/Visualization#VisIt "VisIt"){.wikilink}, a general-purpose 3D visualization tool (a [gallery](https://wci.llnl.gov/simulation/computer-codes/visit/gallery) presents examples from chemistry).

See [Visualization](https://docs.alliancecan.ca/Visualization "Visualization"){.wikilink} for more about producing visualizations on Compute Canada clusters.

#### Other tools {#other_tools}

- [CheMPS2](https://github.com/SebWouters/CheMPS2), a \"library which contains a spin-adapted implementation of the density matrix renormalization group (DMRG) for ab initio quantum chemistry.\"
- [Libxc](http://www.tddft.org/programs/octopus/wiki/index.php/Libxc), a library used in density-functional models.
- [Open3DQSAR](http://open3dqsar.sourceforge.net/?Home), a \"tool aimed at pharmacophore exploration by high-throughput chemometric analysis of molecular interaction fields.\"
- [Open Babel](https://docs.alliancecan.ca/Open_Babel "Open Babel"){.wikilink}, a set of tools to enable one \"to search, convert, analyze, or store data from molecular modeling, chemistry, solid-state materials, biochemistry, or related areas.\"
- [PCMSolver](https://pcmsolver.readthedocs.org), a tool for code development related to the Polarizable Continuum Model. Some applications listed above offer built-in capabilities related to the PCM.
- [RDKit](https://docs.alliancecan.ca/RDKit "RDKit"){.wikilink}, a collection of cheminformatics and machine-learning software written in C++ and Python.
- [Spglib](https://github.com/atztogo/spglib), a library for development relating to the symmetry of crystals.
