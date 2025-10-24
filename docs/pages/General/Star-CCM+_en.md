---
title: "Star-CCM+/en"
url: "https://docs.alliancecan.ca/wiki/Star-CCM%2B/en"
category: "General"
last_modified: "2025-10-21T18:37:03Z"
page_id: 5428
display_title: "Star-CCM+"
---

`<languages />`{=html}

[STAR-CCM+](https://mdx.plm.automation.siemens.com/star-ccm-plus) is a multidisciplinary engineering simulation suite to model acoustics, fluid dynamics, heat transfer, rheology, multiphase flows, particle flows, solid mechanics, reacting flows, electrochemistry, and electromagnetics. It is developed by Siemens.

# License limitations {#license_limitations}

We have the authorization to host STAR-CCM+ binaries on our servers, but we don\'t provide licenses. You will need to have your own license in order to use this software. A remote POD license can be purchased directly from [Siemens](https://www.plm.automation.siemens.com/global/en/buy/). Alternatively, a local license hosted at your institution can be used, providing it can be accessed through the firewall from the cluster where jobs are to be run.

## Configuring your account {#configuring_your_account}

To configure your account to use a license server with the Star-CCM+ module, create a license file `$HOME/.licenses/starccm.lic` with the following layout:

where `<server>`{=html} and `<port>`{=html} should be changed to specify the hostname (or ip address) and the static vendor port of the license server respectively. Note that manually setting `CDLMD_LICENSE_FILE` equal to `<port>`{=html}@`<server>`{=html} in your slurm script is not required; Instead, when a Star-CCM+ module is loaded this variable is automatically set to your `<i>`{=html}\$HOME/.licenses/starccm.lic`</i>`{=html} file.

### POD license file {#pod_license_file}

Researchers with a POD license purchased from [Siemens](https://www.plm.automation.siemens.com/global/en/buy/) must manually set the `LM_PROJECT` environment variable equal to `<I>`{=html}YOUR CD-ADAPCO PROJECT ID`</i>`{=html} in your slurm script. Also the `~/.licenses/starccm.lic` file should be configured as follows on each cluster:

# Cluster batch job submission {#cluster_batch_job_submission}

When submitting jobs on a cluster for the first time, you must set up the environment to use your license. If you are using Siemans remote `<i>`{=html}pay-on-usage`</i>`{=html} license server then create a `~/.licenses/starccm.lic` file as shown in the `<b>`{=html}Configuring your account- POD license file`</b>`{=html} section above and license checkouts should immediately work. If however you are using an institutional license server, then after creating your `~/.licenses/starccm.lic` file you must also submit a problem ticket to [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink} so we can help co-ordinate the necessary one time network firewall changes required to access it (assuming the server has never been setup to be accessed from the Alliance cluster you will be using). If you still have problems getting the licensing to work then try removing or renaming file `~/.flexlmrc` since previous search paths and/or license server settings maybe stored in it. Note that temporary output files from starccm jobs runs may accumulate in hidden directories named `~/.star-version_number` consuming valuable quota space. These can be removed by periodically running `rm -ri ~/.starccm*` and replying yes when prompted.

## Slurm scripts {#slurm_scripts}

`<tabs>`{=html} `<tab name="Fir/Rorqual/Nibi" >`{=html}

`awk '{print $2}')`

port=\$(cat \$CDLMD_LICENSE_FILE grep -Eo \'\[0-9\]+\$\') nmap \$server -Pn -p \$port grep -v \'\^\$\'; echo

slurm_hl2hl.py \--format STAR-CCM+ \> \$SLURM_TMPDIR/machinefile NCORE=\$((SLURM_NNODES \* SLURM_CPUS_PER_TASK \* SLURM_NTASKS_PER_NODE))

if \[ -n \"\$LM_PROJECT\" \]; then

`  # Siemens PoD license server`\
`  starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -power -podkey $LM_PROJECT -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE -mpi intel -fabric psm2`

else

`  # Institutional license server`\
`  starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE -mpi intel -fabric psm2`

fi }} `</tab>`{=html} `<tab name="Narval" >`{=html}

`</tab>`{=html} `<tab name="Trillium" >`{=html}

`sleep 5`\
`         echo "Attempt number: "$I`\
`         if [ -n "$LM_PROJECT" ]; then`\
`         # Siemens PoD license server`\
`         starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -power -podkey $LM_PROJECT -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE`\
`       else`\
`         # Institutional license server`\
`         starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE`\
`       fi`\
`       RET=$?`\
`       i=$((i+1))`

done exit \$RET }} `</tab>`{=html} `</tabs>`{=html}

# Graphical use {#graphical_use}

To run starccm+ in graphical mode it is recommended to use an [OnDemand](https://docs.alliancecan.ca/wiki/Nibi#Access_through_Open_OnDemand_(OOD)) or JupyterLab system to start a remote desktop. In addition to configuring `~/.licenses/starccm.lic` research groups with a POD license should also run `export LM_PROJECT='CD-ADAPCO PROJECT ID'`and optionally append `<b>`{=html}-power`</b>`{=html} to the list of command `starccm+` line options below. Note that `module avail starccm-mixed` will display which starccm versions are available within the StdEnv/version that you currently have loaded. Alternatively running `module spider starccm-mixed` will show all available starccm module versions available within all StdEnv module versions.

## OnDemand

1\. Connect to an OnDemand system using one of the following URLs in your laptop browser :\
[NIBI](https://docs.alliancecan.ca/wiki/Nibi#Access_through_Open_OnDemand_(OOD)): [`https://ondemand.sharcnet.ca`](https://ondemand.sharcnet.ca)

`FIR: `[`https://jupyterhub.fir.alliancecan.ca`](https://jupyterhub.fir.alliancecan.ca)\
`RORQUAL: `[`https://jupyterhub.rorqual.alliancecan.ca`](https://jupyterhub.rorqual.alliancecan.ca)\
`TRILLIUM: `[`https://ondemand.scinet.utoronto.ca`](https://ondemand.scinet.utoronto.ca)

2\. Open a new terminal window in your desktop and run one of:

:   `<b>`{=html}STAR-CCM+ 18.04.008 (or newer versions)`</b>`{=html}

    :   `module load StdEnv/2023` (default)
    :   `module load starccm-mixed/18.04.008` \*\*OR\*\* `starccm/18.04.008-R8`
    :   starccm+ -rr server
:   `<b>`{=html}STAR-CCM+ 15.04.010`</b>`{=html} \--\> `<b>`{=html}18.02.008 (version range)`</b>`{=html}

    :   `module load CcEnv StdEnv/2020`
    :   `module load starccm-mixed/15.04.010` \*\*OR\*\* `starccm/15.04.010-R8`
    :   starccm+ -mesa
:   `<b>`{=html}STAR-CCM+ 13.06.012 (or older versions)`</b>`{=html}

    :   `module load CcEnv StdEnv/2016`
    :   `module load starccm-mixed/13.06.012` \*\*OR\*\* `starccm/13.06.012-R8`
    :   starccm+ -mesa

## VncViewer

1\. Connect with a VncViewer client to a login or compute node by following [TigerVNC](https://docs.alliancecan.ca/VNC "TigerVNC"){.wikilink}\
2. Open a new terminal window in your desktop and run one of:

:   `<b>`{=html}STAR-CCM+ 15.04.010 (or newer versions)`</b>`{=html}

    :   `module load StdEnv/2020`
    :   `module load starccm-mixed/17.02.007` \*\*OR\*\* `starccm/17.02.007-R8`
    :   starccm+
:   `<b>`{=html}STAR-CCM+ 14.06.010, 14.04.013, 14.02.012`</b>`{=html}

    :   `module load StdEnv/2016`
    :   `module load starccm-mixed/14.06.010` \*\*OR\*\* `starccm/14.06.010-R8`
    :   starccm+
:   `<b>`{=html}STAR-CCM+ 13.06.012 (or older versions)`</b>`{=html}

    :   `module load StdEnv/2016`
    :   `module load starccm-mixed/13.06.012` \*\*OR\*\* `starccm/13.06.012-R8`
    :   starccm+ -mesa
