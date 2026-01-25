---
title: "AnsysEDT"
url: "https://docs.alliancecan.ca/wiki/AnsysEDT"
category: "General"
last_modified: "2026-01-14T18:05:11Z"
page_id: 28712
display_title: "AnsysEDT"
---

AnsysEDT bundles electromagnetics simulation solutions such as Ansys HFSS, Ansys Maxwell, Ansys Q3D Extractor, Ansys SIwave, and Ansys Icepak using electrical CAD (ECAD) and mechanical CAD (MCAD) workflows.  AnsysEDT also integrates with the complete Ansys portfolio of thermal, fluid, and mechanical solvers for comprehensive multiphysics analysis.

= Licensing =

The Alliance is a hosting provider for AnsysEDT. This means we have the software installed on our clusters, but do not provide a generic license accessible to everyone. However, many institutions, faculties, and departments already have license servers that can be used if the legal aspects can be worked out.  Network changes would need to be made to enable the license server to be reached from the cluster compute nodes.  The Ansys software would then be able to check out licenses after loading the ansysedt module.  For help contact technical support.

== Configuring your license file ==
Specify your ansysedt license server by creating a file named $HOME/.licenses/ansys.lic consisting of two lines.  See Configuring your license file on the ansys wiki page for further details.

= Cluster batch job submission =

AnsysEDT can be run interactively in batch (non-gui) mode by first starting an salloc session with options salloc --time=3:00:00 --tasks=8 --mem=16G --account=def-account and then copy-pasting the full ansysedt command found in the last line of script-local-cmd.sh, being sure to manually specify $YOUR_AEDT_FILE.

=== Slurm scripts ===

Jobs may be submitted to a cluster queue with the sbatch script-name.sh command using either of the following single node scripts.  Please note these scripts are generic and may require modifications on various clusters.  Before using them, specify the simulation time, memory, number of cores and replace YOUR_AEDT_FILE with your input file name.   A full listing of command line options can be obtained by starting AnsysEDT in graphical mode with commands ansysedt -help or ansysedt -Batchoptionhelp to obtain scrollable graphical popups.

= Graphical use =

Ansys programs may be run interactively in GUI mode on any cluster compute nodes or On Demand systems.

== Compute nodes ==

AnsysEDT  can be run interactively on a single compute node for up to 24 hours.  This approach is ideal for testing large simulations, since all cores and memory can be requested with salloc as described in TigerVNC.  Once connected with vncviewer, any of the following program versions can be started after loading the required modules as shown below.

::: Start an interactive session using the following form of the salloc command (to specify cores and available memory):
::: salloc --time=3:00:00 --nodes=1 --cores=8 --mem=16G --account=def-group
::: xfwm4 --replace & (then hit enter twice)
::: module load StdEnv/2023 ansysedt/2023R2, or
::: module load StdEnv/2023 ansysedt/2024R2.1
::: ansysedt
::: o Click Tools -> Options -> HPC and Analysis Options -> Edit then :
:::: 1) untick Use Automatic Settings box (required one time only)
:::: 2) under Machines tab do not change Cores (auto-detected from slurm)
::: o To run interactive analysis click:  Project -> Analyze All

== OnDemand ==

To run starccm+ in graphical interactive mode on a remote desktop it is recommended to use an OnDemand or JupyterLab system as follows:

1. Connect to an OnDemand system using one of the following URLs in your laptop browser :
 NIBI: https://ondemand.sharcnet.ca
 FIR: https://jupyterhub.fir.alliancecan.ca
 RORQUAL: https://jupyterhub.rorqual.alliancecan.ca
 TRILLIUM: https://ondemand.scinet.utoronto.ca
2. Open a new terminal window in your desktop and run:
::: module load StdEnv/2023  (default)
::: module load ansysedt/2024R2.1 **OR** ansysedt/2023R2
::: Type ansysedt in the terminal and wait for the gui to start
::: The following only needs to be done once:
:::: click Tools -> Options -> HPC and Analysis Options -> Options
:::: change HPC License pulldown to Pool (allows > 4 cores to be used)
:::: click OK
3.  To run the 2024R2.1 Antennas example copy the corresponding version into your account:
:::: module load ansysedt/2024R2.1
:::: mkdir -p ~/Ansoft/$EBVERSIONANSYSEDT; cd ~/Ansoft/$EBVERSIONANSYSEDT; rm -rf Antennas
:::: cp -a $EBROOTANSYSEDT/v242/Linux64/Examples/HFSS/Antennas ~/Ansoft/$EBVERSIONANSYSEDT
4. Now to run the example:
:::: open a simulation .aedt file then click HFSS -> Validation Check
:::: (if errors are reported by the validation check, close then reopen the simulation and repeat as required)
::::  to run simulation click Project -> Analyze All
:::: to quit without saving the converged solution click File -> Close -> No
::: If the program crashes and won't restart try running the following commands:
:::: pkill -9 -u $USER -f "ansys*|mono|mwrpcss|apip-standalone-service"
:::: rm -rf ~/.mw (ansysedt will re-run first-time configuration on startup)

= Site-Specific =

== SHARCNET license ==

The usage terms of the SHARCNET ANSYS License (which includes AnsysEDT) along with other various details maybe found in the SHARCNET license section of the Ansys wiki and will not be repeated here.

==== License file ====

The SHARCNET Ansys license can be used for the AnsysEDT modules on any Alliance cluster by any researcher for free, by configuring your ansys.lic file as follows:

[username@cluster:~] cat ~/.licenses/ansys.lic
setenv("ANSYSLMD_LICENSE_FILE", "1055@license3.sharcnet.ca")
setenv("ANSYSLI_SERVERS", "2325@license3.sharcnet.ca")