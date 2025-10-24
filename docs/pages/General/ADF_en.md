---
title: "ADF/en"
url: "https://docs.alliancecan.ca/wiki/ADF/en"
category: "General"
last_modified: "2022-07-19T13:25:01Z"
page_id: 5365
display_title: "ADF"
---

`<languages />`{=html}

## Introduction

**Important:** ADF has been renamed to AMS since the 2020 version. Significant changes such as the input and output formats have been made in the new AMS. Please refer to [AMS](https://docs.alliancecan.ca/AMS "AMS"){.wikilink} for more information.

The [SCM (Software for Chemistry and Materials) Amsterdam Modeling Suite](https://www.scm.com/) originally the ADF (Amsterdam Density Functional) Modeling Suite, offers powerful computational chemistry tools for many research areas such as homogeneous and heterogeneous catalysis, inorganic chemistry, heavy element chemistry, various types of spectroscopy, and biochemistry.

Compute Canada users have access to the following products:

- ADF
- ADF-GUI
- BAND
- BAND-GUI
- DFTB
- ReaxFF
- COSMO-RS
- QE-GUI
- NBO6

## Running SCM on Graham {#running_scm_on_graham}

The `adf` module is installed only on [Graham](https://docs.alliancecan.ca/Graham "Graham"){.wikilink} due to license restrictions. To check what versions are available use the `module spider` command as follows:

`[name@server $] module spider adf`

For module commands, please see [Using modules](https://docs.alliancecan.ca/Utiliser_des_modules/en "Using modules"){.wikilink}.

### Job submission {#job_submission}

Graham uses the Slurm scheduler; for details about submitting jobs, see [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink}.

#### Single ADF or BAND run {#single_adf_or_band_run}

This mysub.sh script is for a whole-node job. The last two lines load version 2019.305 and call ADF directly.

This is the input file used in the script:

#### Multiple ADF or BAND runs {#multiple_adf_or_band_runs}

Multiple calculations can be combined into a single job by creating a input file such as this:

The following slurm script is identical to the one used for a single run (mysub.sh), except the last line calls the GO_H2O.run script, instead of ADF.

### Examples

Example input/output for ADF can be found on Graham under

`/home/jemmyhu/tests/test_ADF/2019.305/test_adf/`

The same procedure applies to BAND jobs, see band_test.inp and band_test.sh examples under

`/home/jemmyhu/tests/test_ADF/2019.305/test_band/`

## Running SCM-GUI {#running_scm_gui}

Rendering over an SSH connection with X11 forwarding is very slow for GUI applications such as ADF-GUI. We recommend you use [VNC](https://docs.alliancecan.ca/VNC "VNC"){.wikilink} to connect if you will be running ADF-GUI.

### Graham

ADF can be run interactively in graphical mode on a Graham compute node (3hr time limit) over TigerVNC with these steps:

1.  [Install a TigerVNC](https://docs.alliancecan.ca/VNC#Setup "Install a TigerVNC"){.wikilink} client on your desktop
2.  [Connect](https://docs.alliancecan.ca/VNC#Compute_Nodes "Connect"){.wikilink} to a compute node with vncviewer
3.  `module load adf`
4.  `adfinput`

### Gra-vdi {#gra_vdi}

Adf can be run interactively in graphical mode on gra-vdi (no connection time limit) over TigerVNC with these steps:

1.  [Install a TigerVNC](https://docs.alliancecan.ca/VNC#Setup "Install a TigerVNC"){.wikilink} client on your desktop
2.  [Connect](https://docs.alliancecan.ca/VNC#VDI_Nodes "Connect"){.wikilink} to gra-vdi.computecanada.ca with vncviewer
3.  `module load clumod`
4.  `module load adf`
5.  `adfinput`

A tutorial pdf showing how to install, connect and run ADF-GUI using TigerVNC on gra-vdi can be found [here](https://www.sharcnet.ca/~jemmyhu/TigerVNC-for-ADF-GUI.pdf).

### Locally

SCM has a separate license to run ADF-GUI on a local desktop machine. If you are interested contact <license@scm.com> to purchase your own license.
