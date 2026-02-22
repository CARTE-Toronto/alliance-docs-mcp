---
title: "AnsysEDT"
url: "https://docs.alliancecan.ca/wiki/AnsysEDT"
category: "General"
last_modified: "2026-02-13T15:36:34Z"
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

To run starccm+ in graphical mode use an  OnDemand or JupyterLab system to start a remote desktop as follows:

== OnDemand ==

1. Connect to an OnDemand system using one of the following URLs in your laptop browser :
 NIBI: https://ondemand.sharcnet.ca
 FIR: https://jupyterhub.fir.alliancecan.ca
 NARVAL:  https://portail.narval.calculquebec.ca/
 RORQUAL: https://jupyterhub.rorqual.alliancecan.ca
 TRILLIUM: https://ondemand.scinet.utoronto.ca
2. Open a new terminal window in your desktop and run:
::: module load StdEnv/2023  (default)
::: module load ansysedt/2024R2.1 **OR** ansysedt/2023R2
::: Type ansysedt in the terminal and wait for the gui to start
3. Verify the following settings
::: The following only needs to be done once:
:::: click Tools -> Options -> HPC and Analysis Options -> Edit
:::: When the Analysis Configuration panel appears untic Use Automatic Settings
:::: Ensure the settings in the Machine tab correspond requested desktop resources such as:
::::  | Tasks 1 | Cores 4 | Allocated_Cores | GPUs 0 | RAM 90 | tic Enabled |
:::: Click the OK button to save any changes and close the Analysis Configuration panel
:::: Click the OK button close the HPC and Analysis Options panel
4.  To retrieve the 2024R2.1 Antennas examples, copy its directory under your account as follows:
:::: module load ansysedt/2024R2.1
:::: mkdir -p ~/Ansoft/$EBVERSIONANSYSEDT; rm -rf ~/Ansoft/$EBVERSIONANSYSEDT/Antennas
:::: cp -a $EBROOTANSYSEDT/v242/Linux64/Examples/HFSS/Antennas ~/Ansoft/$EBVERSIONANSYSEDT
5. Now to run the example:
:::: Open one of the Antennas examples .aedt files then click HFSS -> Validation Check
:::: Click simulation -> setup -> advanced -> Mesh/Solution options -> Use Defaults
:::: Start simulation running by clicking Simulation -> Analyze All
:::: To quit without saving the converged solution click File -> Close -> No
6. If ansysedt crashes and won't restart try running the following commands:
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