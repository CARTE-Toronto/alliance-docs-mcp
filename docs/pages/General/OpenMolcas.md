---
title: "OpenMolcas"
url: "https://docs.alliancecan.ca/wiki/OpenMolcas"
category: "General"
last_modified: "2023-12-01T18:13:42Z"
page_id: 24675
display_title: "OpenMolcas"
---

## Introduction

[OpenMolcas](https://github.com/Molcas/OpenMolcas) is a quantum chemistry software package. It includes programs to apply many different electronic structure methods to chemical systems, but its key feature is the multiconfigurational approach, with methods like CASSCF and CASPT2.

OpenMolcas is not a fork or reimplementation of Molcas, it is a large part of the Molcas codebase that has been released as free and open-source software (FOSS) under the Lesser General Public License (LGPL). Some parts of Molcas remain under a different license by decision of their authors (or impossibility to reach them), and are therefore not included in OpenMolcas.

## Running OpenMolcas on national clusters {#running_openmolcas_on_national_clusters}

The `OpenMolcas` module is installed on national clusters. To check what versions are available use the `module spider` command as follows:

`[name@server $] module spider openmolcas`

For module commands, please see [Using modules](https://docs.alliancecan.ca/Utiliser_des_modules/en "Using modules"){.wikilink}.

### Job submission {#job_submission}

The national clusters use the Slurm scheduler; for details about submitting jobs, see [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink}.

#### Example script for serial OpenMolcas job {#example_script_for_serial_openmolcas_job}

Currently only serial job works for OpenMolcas. OpenMolcas\'s parallel implementation uses Global Array which does not work on the clusters.

#### Example input file for PrO2.inp {#example_input_file_for_pro2.inp}
