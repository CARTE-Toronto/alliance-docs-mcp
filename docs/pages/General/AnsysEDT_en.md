---
title: "AnsysEDT/en"
url: "https://docs.alliancecan.ca/wiki/AnsysEDT/en"
category: "General"
last_modified: "2025-09-04T19:06:52Z"
page_id: 28884
display_title: "AnsysEDT"
---

`<languages />`{=html}

[AnsysEDT](https://www.ansys.com/products/electronics) bundles electromagnetics simulation solutions such as Ansys HFSS, Ansys Maxwell, Ansys Q3D Extractor, Ansys SIwave, and Ansys Icepak using electrical CAD (ECAD) and mechanical CAD (MCAD) workflows. AnsysEDT also integrates with the complete Ansys portfolio of thermal, fluid, and mechanical solvers for comprehensive multiphysics analysis.

# Licensing

The Alliance is a hosting provider for AnsysEDT. This means we have the software installed on our clusters, but do not provide a generic license accessible to everyone. However, many institutions, faculties, and departments already have license servers that can be used if the legal aspects can be worked out. Network changes would need to be made to enable the license server to be reached from the cluster compute nodes. The Ansys software would then be able to check out licenses after loading the ansysedt module. For help contact [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink}.

## Configuring your license file {#configuring_your_license_file}

Specify your ansysedt license server by creating a file named `$HOME/.licenses/ansys.lic` consisting of two lines. See [Configuring your license file](https://docs.alliancecan.ca/Ansys#Configuring_your_license_file "Configuring your license file"){.wikilink} on the ansys wiki page for further details.

# Cluster batch job submission {#cluster_batch_job_submission}

AnsysEDT can be run interactively in batch (non-gui) mode by first starting an salloc session with options `salloc --time=3:00:00 --tasks=8 --mem=16G --account=def-account` and then copy-pasting the full `ansysedt` command found in the last line of `<i>`{=html}script-local-cmd.sh`</i>`{=html}, being sure to manually specify \$YOUR_AEDT_FILE.

### Slurm scripts {#slurm_scripts}

Jobs may be submitted to a cluster queue with the `sbatch script-name.sh` command using either of the following single node scripts. As of January 2023, the scripts had only been tested on Graham and therefore may be updated in the future as required to support other clusters. Before using them, specify the simulation time, memory, number of cores and replace YOUR_AEDT_FILE with your input file name. A full listing of command line options can be obtained by starting AnsysEDT in [graphical mode](https://docs.alliancecan.ca/ANSYS#Graphical_use "graphical mode"){.wikilink} with commands `ansysedt -help` or `ansysedt -Batchoptionhelp` to obtain scrollable graphical popups.

`<tabs>`{=html} `<tab name="Single node (command line)">`{=html} `{{File
|name=script-local-cmd.sh
|lang="bash"
|contents=
#!/bin/bash

#SBATCH --account=account      # Specify your account (def or rrg)
#SBATCH --time=00-01:00        # Specify time (DD-HH:MM)
#SBATCH --mem=16G              # Specify memory (set to 0 to use all compute node memory)
#SBATCH --ntasks=8             # Specify cores (beluga 40, cedar 32 or 48, graham 32 or 44, narval 64)
#SBATCH --nodes=1              # Request one node (Do Not Change)

#module load StdEnv/2020
#module load ansysedt/2021R2

module load StdEnv/2023
module load ansysedt/2023R2    # or newer

# Uncomment next line to run a test example:
cp -f $EBROOTANSYSEDT/AnsysEM21.2/Linux64/Examples/HFSS/Antennas/TransientGeoRadar.aedt .

# Specify input file such as:
YOUR_AEDT_FILE="TransientGeoRadar.aedt"

# Remove previous output:
rm -rf $YOUR_AEDT_FILE.* ${YOUR_AEDT_FILE}results

# ---- do not change anything below this line ---- #

echo -e "\nANSYSLI_SERVERS= $ANSYSLI_SERVERS"
echo "ANSYSLMD_LICENSE_FILE= $ANSYSLMD_LICENSE_FILE"
echo -e "SLURM_TMPDIR= $SLURM_TMPDIR on $SLURMD_NODENAME\n"

export KMP_AFFINITY=disabled
ansysedt -monitor -UseElectronicsPPE -ng -distributed -machinelist list=localhost:1:$SLURM_NTASKS \
-batchoptions "TempDirectory=$SLURM_TMPDIR HPCLicenseType=pool HFSS/EnableGPU=0" -batchsolve "$YOUR_AEDT_FILE"
}}`{=mediawiki} `</tab>`{=html} `<tab name="Single node (options file)">`{=html} `{{File
|name=script-local-opt.sh
|lang="bash"
|contents=
#!/bin/bash

#SBATCH --account=account      # Specify your account (def or rrg)
#SBATCH --time=00-01:00        # Specify time (DD-HH:MM)
#SBATCH --mem=16G              # Specify memory (set to 0 to allocate all compute node memory)
#SBATCH --ntasks=8             # Specify cores (beluga 40, cedar 32 or 48, graham 32 or 44, narval 64)
#SBATCH --nodes=1              # Request one node (Do Not Change)

#module load StdEnv/2020
#module load ansysedt/2021R2

module load StdEnv/2023
module load ansysedt/2023R2    # or newer

# Uncomment next line to run a test example:
cp -f $EBROOTANSYSEDT/AnsysEM21.2/Linux64/Examples/HFSS/Antennas/TransientGeoRadar.aedt .

# Specify input filename such as:
YOUR_AEDT_FILE="TransientGeoRadar.aedt"

# Remove previous output:
rm -rf $YOUR_AEDT_FILE.* ${YOUR_AEDT_FILE}results

# Specify options filename:
OPTIONS_TXT="Options.txt"

# Write sample options file
rm -f $OPTIONS_TXT
cat > $OPTIONS_TXT <<EOF
\$begin 'Config'
'TempDirectory'='$SLURM_TMPDIR'
'HPCLicenseType'='pool'
'HFSS/EnableGPU'=0
\$end 'Config'
EOF

# ---- do not change anything below this line ---- #

echo -e "\nANSYSLI_SERVERS= $ANSYSLI_SERVERS"
echo "ANSYSLMD_LICENSE_FILE= $ANSYSLMD_LICENSE_FILE"
echo -e "SLURM_TMPDIR= $SLURM_TMPDIR on $SLURMD_NODENAME\n"

export KMP_AFFINITY=disabled

ansysedt -monitor -UseElectronicsPPE -ng -distributed -machinelist list=localhost:1:$SLURM_NTASKS \
-batchoptions $OPTIONS_TXT -batchsolve "$YOUR_AEDT_FILE"
}}`{=mediawiki} `</tab>`{=html} `</tabs>`{=html}

# Graphical use {#graphical_use}

Ansys programs may be run interactively in GUI mode on cluster compute nodes or Graham VDI Nodes.

## Compute nodes {#compute_nodes}

AnsysEDT can be run interactively on a single compute node for up to 24 hours. This approach is ideal for testing large simulations, since all cores and memory can be requested with salloc as described in [TigerVNC](https://docs.alliancecan.ca/VNC#Compute_Nodes "TigerVNC"){.wikilink}. Once connected with vncviewer, any of the following program versions can be started after loading the required modules as shown below.

:   

    :   Start an interactive session using the following form of the salloc command (to specify cores and available memory):
    :   `salloc --time=3:00:00 --nodes=1 --cores=8 --mem=16G --account=def-group`
    :   `xfwm4 --replace &` (then hit enter twice)
    :   `module load StdEnv/2020 ansysedt/2021R2`, or
    :   `module load StdEnv/2020 ansysedt/2023R2`, or
    :   `module load StdEnv/2023 ansysedt/2023R2`, or
    :   `module load StdEnv/2023 ansysedt/2024R2` \<\-\-- !!! this module version is currently only on graham !!!
    :   `ansysedt`
    :   o Click `Tools -> Options -> HPC and Analysis Options -> Edit` then :

        :   1\) untick Use Automatic Settings box (required one time only)
        :   2\) under Machines tab do not change Cores (auto-detected from slurm)
    :   o To run interactive analysis click: `Project -> Analyze All`

## VDI nodes {#vdi_nodes}

Ansys programs can be run for up to 7 days on grahams VDI nodes (gra-vdi.alliancecan.ca) using 8 cores (16 cores max) and 128GB memory. The VDI System provides GPU OpenGL acceleration and is therefore ideal for tasks that benefit from high performance graphics. One might use VDI to create or modify simulation input files, post-process data or visualize simulation results. To log in, connect with [TigerVNC](https://docs.alliancecan.ca/VNC#VDI_Nodes "TigerVNC"){.wikilink} then open a new terminal window and start one of the program versions shown below. The vertical bar `|` notation is used to separate the various commands. The maximum job size for any parallel job run on gra-vdi should be limited to 16 cores to avoid overloading the servers and impacting other users. To run two simultaneous GUI jobs (16 cores max each) on gra-vdi, establish two independent vnc sessions. For the first session, connect to gra-vdi3.sharcnet.ca with vnc. For the second session connect to gra-vdi4.sharecnet.ca also with vnc. Then within each session, start Ansys in GUI mode and run your simulation. Note that simultaneous simulations should in general be run in different directories to avoid file conflict issues. Unlike compute nodes vnc connections (which impose slurm limits through salloc) there is no time limit constraint on gra-vdi when running simulations.

:   

    :   Open a terminal window and load a module:
    :   `module load SnEnv`
    :   `module load ansysedt/2023R2` (or older versions), or,
    :   `module load SnEnv`
    :   `module load ansys/2024R2[.04]`, or,
    :   `module load CcEnv StdEnv/2023`
    :   `module load ansys/2025R1[.02]` (or newer versions, testing)
    :   Type `ansysedt` in the terminal and wait for the gui to start
    :   The following only needs to be done once:

        :   click `Tools -> Options -> HPC and Analysis Options -> Options`
        :   change `HPC License` pulldown to `<b>`{=html}Pool`</b>`{=html} (allows \> 4 cores to be used)
        :   click `OK`
    :   \-\-\-\-\-\-\-\-\-- EXAMPLES \-\-\-\-\-\-\-\-\--
    :   To copy the 2023R2 Antennas examples directory into your account:

        :   login to a cluster such as graham
        :   `module load ansysedt/2023R2`
        :   `mkdir -p ~/Ansoft/$EBVERSIONANSYSEDT; cd ~/Ansoft/$EBVERSIONANSYSEDT; rm -rf Antennas`
        :   `cp -a $EBROOTANSYSEDT/v232/Linux64/Examples/HFSS/Antennas ~/Ansoft/$EBVERSIONANSYSEDT`
    :   To run an example:

        :   open a simulation .aedt file then click `HFSS -> Validation Check`
        :   (if errors are reported by the validation check, close then reopen the simulation and repeat as required)
        :   to run simulation click `Project -> Analyze All`
        :   to quit without saving the converged solution click `File -> Close -> No`
    :   If the program crashes and won\'t restart try running the following commands:

        :   `pkill -9 -u $USER -f "ansys*|mono|mwrpcss|apip-standalone-service"`
        :   `rm -rf ~/.mw` (ansysedt will re-run first-time configuration on startup)

# Site-Specific {#site_specific}

## SHARCNET license {#sharcnet_license}

The usage terms of the SHARCNET ANSYS License (which includes AnsysEDT) along with other various details maybe found in the SHARCNET license section of the Ansys wiki and will not be repeated here.

#### License file {#license_file}

The SHARCNET Ansys license can be used for the AnsysEDT modules on any Alliance cluster by any researcher for free, by configuring your `ansys.lic` file as follows:

``` bash
[username@cluster:~] cat ~/.licenses/ansys.lic
setenv("ANSYSLMD_LICENSE_FILE", "1055@license3.sharcnet.ca")
setenv("ANSYSLI_SERVERS", "2325@license3.sharcnet.ca")
```

## Local modules {#local_modules}

Use of local modules installed on gra-vdi or Graham may be of interest when there is a version available, which is not (yet) installed into the global environment on all systems either because it\'s very new or there are technical issues. When starting programs from local modules on gra-vdi, you can select the CMC license server or accept the default SHARCNET license server as they are furnished with a startup wrapper including reading your `~/.licenses/ansysedt.lic` file. Suitable usage of AnsysEDT on gra-vdi includes: running a single test job interactively with up to 8 cores and/or 128G RAM, create or modify simulation input files, post-process or visualize data. A local ansysedt module can also be loaded on the Graham cluster; however, the procedure is slightly different as will be shown below.

### Use on gra-vdi {#use_on_gra_vdi}

When using gra-vdi, researchers have the choice of loading Ansys modules from our global environment (after loading CcEnv) or loading Ansys modules installed locally on the machine itself (after loading SnEnv). To load a local ansysedt module on gra-vdi and then run the program in graphical mode, follow these steps :

1.  Connect to gra-vdi.computecanada.ca with [TigerVNC](https://docs.alliancecan.ca/VNC#VDI_Nodes "TigerVNC"){.wikilink}.
2.  Open a new terminal window and load a module:

    `module load SnEnv`\
    `module load ansysedt/2024R2.1` (or older)
3.  Start the Ansys Electromagnetics Desktop program by typing the following command: `ansysedt`
4.  Press `y` and `Enter` to accept the conditions.
5.  Press `Enter` to accept the `n` option and use the SHARCNET license server by default (note that `<i>`{=html}\~/.licenses/ansysedt.lic`</i>`{=html} will be used if present, otherwise ANSYSLI_SERVERS and ANSYSLMD_LICENSE_FILE will be used if set in your environment for example to some other remote license server). If you change `n` to `y` and hit enter, the CMC license server will be used.

### Use on Graham {#use_on_graham}

To load a local ansysedt module on graham and then run the program in graphical mode, follow these steps : (coming soon).
