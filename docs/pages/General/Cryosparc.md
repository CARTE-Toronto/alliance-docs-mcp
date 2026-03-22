---
title: "Cryosparc"
url: "https://docs.alliancecan.ca/wiki/Cryosparc"
category: "General"
last_modified: "2026-03-18T21:42:30Z"
page_id: 32795
display_title: "Cryosparc"
---

CryoSPARC is a state of the art scientific software platform for cryo-electron microscopy (cryo-EM) used in research and drug discovery pipelines.

== Installation ==
CryoSPARC is not supported on login nodes and must be installed and run on a GPU node. Running CryoSPARC on a login node can overload shared system resources and may lead to port conflicts when multiple instances are started at the same time.

Currently, only Fir and Nibi support CryoSPARC installation and execution on GPU nodes, as these systems provide outbound HTTPS access from GPU nodes to the CryoSPARC download site.

This tutorial describes how to build a standalone CryoSPARC instance in an Apptainer container image on a GPU node, allowing users to run CryoSPARC through GPU batch jobs while reducing filesystem load.

=== Step 1. Acquire the license ===
Complete the form at https://cryosparc.com/download to request a license. The license key is a string of letters and numbers in the following format:

1a23b4d5-67ef-89g0-hi1j-kl2m3n4o5p6q

=== Step 2. Set up the installation directory ===
Create the directory where you would like to install CryoSPARC.
Set up the LICENSE_ID by replacing 1a23b4d5-67ef-89g0-hi1j-kl2m3n4o5p6q with the license key you obtained from https://cryosparc.com/download in Step 1.

=== Step 3. Download the installation package ===

=== Step 4. Submit a GPU job for installation ===
Launch an interactive job:

Prepare the first definition file. Change the value of $HOST_PATH to match the path you set in Step 2.

Build the first container image using the prepared definition file.

Prepare the second definition file. Update the values according to the comments below.

Build CryoSPARC container image:

=== Step 5. Launch CryoSPARC in a batch GPU job ===
Prepare the running job script. Replace export HOST_PATH=/path/to/cryosparc with the path you set in Step 2.

Submit the job:

After the job starts, follow the instructions in the job log file (for example, slurm-123456.out) to open the CryoSPARC graphical interface in your local browser. Use the interface to set up your CryoSPARC job and launch the worker. Once the CryoSPARC job is complete, check the output files and cancel the Slurm job with scancel; otherwise, the job will continue running until it reaches the walltime limit.