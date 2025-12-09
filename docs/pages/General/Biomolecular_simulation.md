---
title: "Biomolecular simulation/en"
url: "https://docs.alliancecan.ca/wiki/Biomolecular_simulation/en"
category: "General"
last_modified: "2025-02-10T19:07:41Z"
page_id: 5661
display_title: "Biomolecular simulation"
---

`<languages />`{=html}

## General

Biomolecular simulation[^1] is the application of molecular dynamics simulations to biochemical research questions. Processes that can be modeled include, but are not limited to, protein folding, drug binding, membrane transport, and the conformational changes critical to protein function.

While biomolecular simulation could be considered a sub-field of computational chemistry, it is sufficiently specialized that we have a Biomolecular Simulations National Team that supports this area. There is nevertheless some overlap of software tools between the two fields. See [Computational chemistry](https://docs.alliancecan.ca/Computational_chemistry "Computational chemistry"){.wikilink} for an annotated list of available software packages in that area.

## Software Packages {#software_packages}

The following software packages are available on our HPC resources:

- [AMBER](https://docs.alliancecan.ca/AMBER "AMBER"){.wikilink}
- [GROMACS](https://docs.alliancecan.ca/GROMACS "GROMACS"){.wikilink}
- [NAMD](https://docs.alliancecan.ca/NAMD "NAMD"){.wikilink}
- [DL_POLY](http://www.scd.stfc.ac.uk/SCD/44516.aspx)
- [HOOMD-blue](http://glotzerlab.engin.umich.edu/hoomd-blue/)
- [LAMMPS](https://docs.alliancecan.ca/LAMMPS "LAMMPS"){.wikilink}
- [OpenKIM](https://openkim.org/), the Knowledgebase of Interatomic Models
- [OpenMM](https://docs.alliancecan.ca/OpenMM "OpenMM"){.wikilink}
- [PLUMED](https://www.plumed.org), a library for code development related to the calculation of free energy in molecular dynamics simulations. See also [GROMACS](https://docs.alliancecan.ca/GROMACS "GROMACS"){.wikilink}.
- [Rosetta](https://www.rosettacommons.org)
- [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/)
- [VMD](https://docs.alliancecan.ca/VMD "VMD"){.wikilink}

### Python Packages (Python Wheels) {#python_packages_python_wheels}

Our [Wheelhouse](https://docs.alliancecan.ca/Available_Python_wheels "Wheelhouse"){.wikilink} contains a number of Python Wheels that can be installed within a [virtual Python environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual Python environment"){.wikilink} and are useful in the domain of biomolecular simulation/molecular dynamics.

This list contains a selection of useful wheels, but is not to be considered complete:

- [ACPYPE: AnteChamber PYthon Parser interfacE](https://docs.alliancecan.ca/ACPYPE "ACPYPE: AnteChamber PYthon Parser interfacE"){.wikilink} is a tool to generate topologies for chemical compounds.
- [MDAnalysis](https://www.mdanalysis.org/) is an object-oriented Python library to analyze trajectories from molecular dynamics (MD) simulations in many popular formats.
- [MDTraj](http://mdtraj.org/) can also read, write and analyze MD trajectories with only a few lines of Python code with wide MD format support.
- [Biopython](https://biopython.org/) is a set of freely available tools for biological computation.
- [foyer](https://foyer.mosdef.org/) is a package for atom-typing as well as applying and disseminating force fields.
- [mBuild](https://mbuild.mosdef.org/) is a hierarchical, component based molecule builder.
- [mdsynthesis](https://mdsynthesis.readthedocs.io/) is a persistence engine for molecular dynamics data.
- [nglview](http://nglviewer.org/): NGL Viewer is a collection of tools for web-based molecular graphics.
- [ParmEd](http://parmed.github.io/ParmEd/) is a general tool for aiding in investigations of biomolecular systems using popular molecular simulation packages.
- [PyRETIS](https://docs.alliancecan.ca/PyRETIS "PyRETIS"){.wikilink} is a Python library for rare event molecular simulations with emphasis on methods based on transition interface sampling and replica exchange transition interface sampling.

Please check the [list of available wheels](https://docs.alliancecan.ca/Available_Python_wheels "list of available wheels"){.wikilink} and use the [avail_wheels command](https://docs.alliancecan.ca/Python#Listing_available_wheels "avail_wheels command"){.wikilink} on our clusters to see what is available.

If you require additional Python packages or newer versions, please [contact Support](https://docs.alliancecan.ca/Technical_support "contact Support"){.wikilink}.

## Workshops and Training Material {#workshops_and_training_material}

The *Molecular Modelling and Simulation National Team* is offering Molecular Dynamics workshops. Future workshops will be announced in our Newsletters.

The workshop material is also available for self-study:

1.  [Practical considerations for Molecular Dynamics](https://computecanada.github.io/molmodsim-md-theory-lesson-novice/)
2.  [Visualizing Structures with VMD](https://computecanada.github.io/molmodsim-vmd-visualization/)
3.  [Running Molecular Dynamics with Amber on our clusters](https://computecanada.github.io/molmodsim-amber-md-lesson/)
4.  [Analyzing Molecular Dynamics Data with PYTRAJ](https://computecanada.github.io/molmodsim-pytraj-analysis/)

## Performance and benchmarking {#performance_and_benchmarking}

A team at [ACENET](https://www.ace-net.ca/) has created a [Molecular Dynamics Performance Guide](https://mdbench.ace-net.ca/mdbench/) for Alliance clusters. It can help you determine optimal conditions for Amber, GROMACS, NAMD, and OpenMM jobs.

## References

[^1]: Ron O. Dror, Robert M. Dirks, J.P. Grossman, Huafeng Xu, and David E. Shaw. \"Biomolecular Simulation: A Computational Microscope for Molecular Biology.\" *Annual Review of Biophysics*, 41:429-452, 2012. <https://doi.org/10.1146/annurev-biophys-042910-155245>
