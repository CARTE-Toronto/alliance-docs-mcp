---
title: "Moose/en"
url: "https://docs.alliancecan.ca/wiki/Moose/en"
category: "General"
last_modified: "2025-10-27T13:21:01Z"
page_id: 31888
display_title: "Moose/en"
---

`<languages />`{=html} [MOOSE (Multiphysics Object Oriented Simulation Environment)](https://mooseframework.inl.gov/) is an object-oriented C++ finite element framework for the development of tightly coupled multiphysics solvers from Idaho National Laboratory. Moose is not officially maintained on our clusters, so users need to compile the software themselves. The following compiling instructions worked for multiple users on our upgraded systems (using StdEnv/2023 environment):

`module load StdEnv/2023`\
`module load python/3.11.5 scipy-stack/2023b python-build-bundle/2025b petsc/3.23.4 eigen/3.4.0 boost/1.85.0`\
`export MOOSE_JOBS=6 METHODS=opt`\
`git clone --depth 1 `[`https://github.com/idaholab/moose.git`](https://github.com/idaholab/moose.git)\
`cd moose/scripts/`\
`./update_and_rebuild_libmesh.sh --with-xdr-include=$EBROOTGENTOO/include/tirpc --with-glpk-include=$EBROOTGENTOO/include --with-eigen-include=$EBROOTEIGEN/include`\
`./update_and_rebuild_wasp.sh`\
`cd ../test`\
`make -j6`\
`./run_tests -j6`

Let us know by writing to our [Technical support](https://docs.alliancecan.ca/Technical_support "wikilink") if these instructions do not work for you.
