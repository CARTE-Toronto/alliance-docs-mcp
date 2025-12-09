---
title: "R-INLA/en"
url: "https://docs.alliancecan.ca/wiki/R-INLA/en"
category: "General"
last_modified: "2025-07-16T19:13:42Z"
page_id: 28933
display_title: "R-INLA"
---

R-INLA is a package in R that do approximate Bayesian inference for Latent Gaussian Models.

== Installation ==

The installation of the R-INLA package is a bit more complicated than most R packages, as it downloads
other pre-compiled executables that need to be made compatible with our Standard software environment.

The scripts below have been tested with the versions mentioned therein.
Because R will always install the latest versions of packages, the versions of the modules will likely have to be adjusted in the future.

Comments in the script:

* (1) Load required modules.  The same modules have to be loaded in the job script as well.
* (2) Install the R-INLA package and its dependencies
* (3) Install the pre-compiled executables that R-INLA needs
* (4) Patch the pre-compiled executables so that they are compatible with our Standard software environment