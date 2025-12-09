---
title: "Firedrake"
url: "https://docs.alliancecan.ca/wiki/Firedrake"
category: "General"
last_modified: "2025-11-27T12:46:05Z"
page_id: 29047
display_title: "Firedrake"
---

Firedrake is an automated system for the solution of partial differential equations using the finite element method (FEM).

Please note that every release of Firedrake requires a specific version of PETSc and several other modules or Python wheels.

= Installation =

Please note that all modules must be loaded before creating and/or activating the Python virtualenv.

== Firedrake 2025.4.2 ==

2025.2.2  immutabledict
|pip install --no-index  firedrake[check]2025.4.2
}}

The above has been tested with both python/3.13 as well as python/3.12.

= Running jobs =
== Firedrake 2025.4.2 ==

The above has been tested with both python/3.13 as well as python/3.12.

= Optional dependencies =
Firedrake has a number of optional dependencies that can be installed into the virtualenv:

* SLEPc and slepc4py are part of the petsc module and always available.
* netgen: we provide precompiled wheels for ngsPETSc and netgen_mesher.
* PyTorch: since we provide precompiled precompiled wheels for torch.
* Jax: since we provide precompiled precompiled wheels for jax.
* VTK: currently we don't have a module for VTK that supports recent enough versions of Python for Firedrake (Python 3.12 and newer).As a temporary workaround VTK can be installed into the virtualenv with: pip install --no-index --find-links ~stuekero/wheels/vtk vtk==9.4.2 until we install a new VTK module.