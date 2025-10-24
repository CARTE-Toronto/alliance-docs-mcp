---
title: "AMS/en"
url: "https://docs.alliancecan.ca/wiki/AMS/en"
category: "General"
last_modified: "2025-09-23T22:25:24Z"
page_id: 15810
display_title: "AMS"
---

`<languages />`{=html}

## Introduction

AMS (Amsterdam Modeling Suite), originally named [ADF](https://docs.alliancecan.ca/ADF "ADF"){.wikilink} (Amsterdam Density Functional), is the [SCM Software for Chemistry and Materials](https://www.scm.com/). AMS offers powerful computational chemistry tools for many research areas such as homogeneous and heterogeneous catalysis, inorganic chemistry, heavy element chemistry, various types of spectroscopy, and biochemistry.

The full SCM module products are available:

- ADF
- ADF-GUI
- BAND
- BAND-GUI
- DFTB
- ReaxFF
- COSMO-RS
- QE-GUI
- NBO6

## Running AMS on Nibi {#running_ams_on_nibi}

The `ams` module is installed on [Nibi](https://docs.alliancecan.ca/Nibi "Nibi"){.wikilink}. The license is an Academic Computing Center license owned by SHARCNET. You may not use the Software for consulting services nor for purposes that have a commercial nature. To check what versions are available, use the `module spider` command as follows:

`[name@server $] module spider ams`

For module commands, please see [Using modules](https://docs.alliancecan.ca/Utiliser_des_modules/en "Using modules"){.wikilink}.

### Job submission {#job_submission}

The clusters use the Slurm scheduler; for details about submitting jobs, see [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink}.

#### Example scripts for an AMS job {#example_scripts_for_an_ams_job}

This H2O_adf.sh example script is to request 32 CPUs on one node. Please use a reasonable number of CPUs instead of simply running a full-node job on Nibi, unless you have demonstrated that your job can scale efficiently to 192 CPUs.

This is the input file used in the script:

#### Example scripts for a band job {#example_scripts_for_a_band_job}

### Notes

1.  The input for AMS is different from ADF, the previous ADF input file will not run for the new AMS. Some examples can be found in /opt/software/ams/2025.102/examples/
2.  Except the output .log file, other files are all saved in a subdirectory AMS_JOBNAME.results. If AMS_JOBNAME is not defined in the input .run file, the default name is ams.results
3.  The restart file name is ams.rkf instead of the TAPE13 in previous ADF versions

For more usage information, please check the manuals in [SCM Support](https://www.scm.com/support/)

## Running AMS-GUI {#running_ams_gui}

### Nibi

AMS can be run interactively in graphical mode on a Nibi compute node (8hr time limit) via OnDemand with these steps:

1.  Log in to [ondemand.sharcnet.ca](https://ondemand.sharcnet.ca)
2.  Select *Nibi Desktop* from *Compute* on the top
3.  Select your options (select 1 core for visualization purpose, don\'t select *Enable VirtualGL*) and press *Launch*
4.  Select *Launch Nibi Desktop* once your job starts
5.  Right click on the desktop and pick *Open it Terminal*
6.  Pick *MATE Terminal* from the *System Tools* menu under the *Applications* menu
7.  `module unload openmpi`
8.  `module load ams`
9.  `amsinput &`(to make AMS input)
10. `amsview &`(for AMS result visualization)

If you need to select *Enable VirtualGL* for some other program that you are using, you will have to disable it for just AMS by starting it with `LD_PRELOAD= amsinput`.

OnDemand Nibi Desktop is intended for AMS-GUI applications, such as making input files and visualizing results. Please do not use it to run regular jobs or long interactive jobs. Select a single core and reasonable memory and runtime.
