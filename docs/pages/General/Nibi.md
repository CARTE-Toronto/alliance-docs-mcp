---
title: "Nibi/en"
url: "https://docs.alliancecan.ca/wiki/Nibi/en"
category: "General"
last_modified: "2025-12-04T17:06:02Z"
page_id: 27510
display_title: "Nibi"
---

`<languages />`{=html}

  ---------------------------------------------------------------------------------------------------------------------------
  Availability: since 31 July 2025
  SSH login node: nibi.alliancecan.ca
  Web interface: [ondemand.sharcnet.ca](https://ondemand.sharcnet.ca)
  Globus collection: [alliancecan#nibi](https://app.globus.org/file-manager?origin_id=07baf15f-d7fd-4b6a-bf8a-5b5ef2e229d3)
  Data transfer node (rsync, scp, sftp,\...): use login nodes
  Portal: [portal.nibi.sharcnet.ca](https://portal.nibi.sharcnet.ca)
  ---------------------------------------------------------------------------------------------------------------------------

Nibi, the Anishinaabemowin word for water, is a general purpose cluster of 134,400 CPU cores and 288 H100 NVIDIA GPUs. Built by [Hypertec](https://www.hypertec.com/), the cluster is hosted and operated by [SHARCNET](https://www.sharcnet.ca/) at University of Waterloo.

# Storage

Parallel storage: 25 petabytes, all [SSD](https://en.wikipedia.org/wiki/Solid-state_drive) from [VAST Data](https://www.vastdata.com/) for `/home`, `/project` and `/scratch`.

Note that Vast implements space accounting for quotas differently: you are \"charged\" for the apparent size of your files. This is in contrast to some Lustre configurations, which transparently compress files, and charge for the space used after compression.

Note also that Nibi is using a new, experimental mechanism for handling /scratch. As on all systems, you have a soft and hard limit, but on Nibi, the soft limit is low (1TB), and you have a 60d grace period. After the grace period expires, the soft limit is enforced (no further file creation/expansion). To fix this, your usage must drop below the soft limit.

# Interconnect fabric {#interconnect_fabric}

- Nokia 200/400G ethernet
  - 200 Gbit/s network bandwidth for CPU nodes.
  - 200 Gbit/s non-blocking network bandwidth between all Nvidia GPU nodes.
  - 200 Gbit/s network bandwidth between all AMD GPU nodes.
  - 24x100 Gbit/s connection to the VAST storage nodes.
  - 2:1 blocking at 400 Gbit/s uplinks for all compute nodes.

The topology of the network is described in the file:

`/opt/software/slurm/24.11.6/etc/topology.conf`

For better performance of tightly coupled multi-node jobs, you may constrain them to use only one network switch, by adding this option to your job submission script:

`#SBATCH --switches=1`

# Node characteristics {#node_characteristics}

  nodes   cores   available memory    node-local storage   CPU                                         GPU
  ------- ------- ------------------- -------------------- ------------------------------------------- ---------------------------------------------------------------------------------------
  700     192     748G or 766000M     3T                   2 x Intel 6972P @ 2.4 GHz, 384MB cache L3   
  10      192     6000G or 6144000M   3T                   2 x Intel 6972P @ 2.4 GHz, 384MB cache L3   
  36      112     2000G or 2048000M   11T                  2 x Intel 8570 @ 2.1 GHz, 300MB cache L3    8 x Nvidia H100 SXM (80 GB)
  6       96      495G or 507000M     3T                   4 x AMD MI300A @ 2.1GHz (Zen4+CDNA3)        The CPU cores and CDNA3-based GPUs are in the same socket and share a unified memory.

# Site specifics {#site_specifics}

## Internet access {#internet_access}

All nodes on Nibi have Internet access, no special firewall permission or proxying is necessary.

## Project space {#project_space}

User directories are no longer created by default on `/project`. User\'s can always create their own directories in the group\'s `/project` using `mkdir`. This allows groups to decide how their `/project` is organized for sharing data amongst group members.

## Scratch quota {#scratch_quota}

An 1 TB soft quota on scratch applies to each user. This soft quota can be exceeded for up to 60 days after which no additional files may be written to scratch. Files may be written again once the user has removed or deleted enough files to bring their total scratch use under 1 TB. See the [Storage and file management](https://docs.alliancecan.ca/Storage_and_file_management "Storage and file management"){.wikilink} for more information.

## Access through Open OnDemand (OOD) {#access_through_open_ondemand_ood}

One can now access the Nibi cluster simply through a web browser. Nibi uses Open OnDemand (OOD), a web based platform that simplifies cluster access by providing a web interface to the login nodes and a remote desktop environment. To login to Nibi, go to <https://ondemand.sharcnet.ca/>, sign in with MFA, you will see a user friendly interface offering options to open a Bash shell terminal or launch a remote desktop session.

## Use of JupyterLab via OOD {#use_of_jupyterlab_via_ood}

![](nibi-jupyterlab.png "nibi-jupyterlab.png") You can run JupyterLab interactively via the Nibi Open OnDemand [portal](https://ondemand.sharcnet.ca).

**Option 1**: working with a pre-configured environment, same as from [JupyterHub](https://docs.alliancecan.ca/JupyterHub "JupyterHub"){.wikilink}

After logging in to the Nibi Open OnDemand [portal](https://ondemand.sharcnet.ca), click "Compute Node" from the top menu and select "Nibi JupyterLab." This will open a page with a form where you can request a new Nibi JupyterLab session.

After completing the form with your requirement details, click "Launch" to submit your request. Once the status of the requested Nibi JupyterLab changes to Running, click "Connect to Jupyter" to open JupyterLab in your web browser.

More details about the pre-configured JupyterLab are described [here](https://docs.alliancecan.ca/JupyterLab#The_JupyterLab_interface "here"){.wikilink}.

**Option 2**: working with a self-built [Python virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "Python virtual environment"){.wikilink}

After logging in to the Nibi Open OnDemand [portal](https://ondemand.sharcnet.ca), click "Compute Node" from the top menu and select "Compute Desktop." This will open a page with a form where you can request a new Compute Desktop session. ![](nibi-desktop.png "nibi-desktop.png")

After completing the form with your requirement details, click "Launch" to submit your request. Once the status of the requested Compute desktop changes to Running, click "Launch Compute Desktop" to connect to the desktop. A Linux desktop will appear.

On the Compute desktop, right-click the mouse in any blank area, a shortcut menu appears; select \"Open in Terminal\" to open a terminal window, where you can create or activate your Python virtual environment that has JupyterLab installed.

If you do not have JupyterLab installed in the Python virtual environment, which you would like to work with, you can have it installed with the command:

Then, you can launch JupyterLab from your Python virtual environment with the command:

You will see JupyterLab is opened in the web browser on the Desktop with your \$HOME contents listed in the file browser panel on JupyterLab.

## Support for VDI via OOD {#support_for_vdi_via_ood}

Nibi no longer offers Virtual Desktop Infrastructure (VDI). Instead, it provides a remote desktop environment through the [portal](https://ondemand.sharcnet.ca/) of Open OnDemand (OOD), offering improved hardware performance and software support.
