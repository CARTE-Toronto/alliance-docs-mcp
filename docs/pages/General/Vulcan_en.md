---
title: "Vulcan/en"
url: "https://docs.alliancecan.ca/wiki/Vulcan/en"
category: "General"
last_modified: "2025-08-09T15:27:02Z"
page_id: 28485
display_title: "Vulcan"
---

`<languages />`{=html}

  ---------------------------------------------------------------------------------------------------------------------------
  Availability: April 15, 2025
  Login node: `<b>`{=html}vulcan.alliancecan.ca`</b>`{=html}
  Globus collection: [Vulcan Globus v5](https://app.globus.org/file-manager?origin_id=97bda3da-a723-4dc0-ba7e-728f35183b43)
  System Status Page: <https://status.alliancecan.ca/system/Vulcan>
  Portal: <https://portal.vulcan.alliancecan.ca>
  ---------------------------------------------------------------------------------------------------------------------------

`<b>`{=html}Vulcan`</b>`{=html} is a cluster dedicated to the needs of the Canadian scientific Artificial Intelligence community. `<b>`{=html}Vulcan`</b>`{=html} is located at the [University of Alberta](https://www.ualberta.ca/) and is managed by the University of Alberta and [Amii](https://amii.ca/). It is named after the town [Vulcan, AB](https://en.wikipedia.org/wiki/Vulcan,_Alberta), located in southern Alberta.

This cluster is part of the Pan-Canadian AI Compute Environment (PAICE).

## Site-specific policies {#site_specific_policies}

Internet access is not generally available from the compute nodes. A globally available Squid proxy is enabled by default with certain domains whitelisted. Contact [technical support](https://docs.alliancecan.ca/Technical_support "technical support"){.wikilink} if you are not able to connect to a domain and we will evaluate whether it belongs on the whitelist.

Maximum duration of jobs is 7 days.

Vulcan is currently open to Amii affiliated PIs with CCAI Chairs. Further access will be announced at a later date.

## Access

To access the Vulcan cluster, each researcher must first [request access in CCDB](https://ccdb.alliancecan.ca/me/access_services).

If you are a PI and need to sponsor other researchers you will have to add them to your AIP RAP. Follow these steps to manage users:

- Go to the \"Resource Allocation Projects\" table on the [CCDB home page](https://ccdb.alliancecan.ca).
- Locate the RAPI of your AIP project (with the `aip-` prefix) and click on it to reach the RAP management page.
- At the bottom of the RAP management page, click on \"Manage RAP memberships.\"
- Enter the CCRI of the user you want to add in the \"Add Members\" section.

## Vulcan hardware specifications {#vulcan_hardware_specifications}

  Nodes   Model         CPU                         Cores   System Memory   GPUs per node          Total GPUs
  ------- ------------- --------------------------- ------- --------------- ---------------------- ------------
  205     Dell R760xa   2 x Intel Xeon Gold 6448Y   64      512 GB          4 x NVIDIA L40s 48GB   820

## Storage system {#storage_system}

`<b>`{=html}Vulcan`</b>`{=html}\'s storage system uses a combination of NVMe flash and HDD storage running on the Dell PowerScale platform with a total usable capacity of approximately 5PB. Home, Scratch, and Project are on the same Dell PowerScale system.

+----------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `<b>`{=html}Home space`</b>`{=html}    | - Location of /home directories.                                                                                                                                                                                                                                                                           |
|                                        | - Each /home directory has a small fixed [quota](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "quota"){.wikilink}.                                                                                                                                                                           |
|                                        | - Not allocated via [RAS](https://alliancecan.ca/en/services/advanced-research-computing/accessing-resources/rapid-access-service) or [RAC](https://alliancecan.ca/en/services/advanced-research-computing/accessing-resources/resource-allocation-competition). Larger requests go to the /project space. |
|                                        | - Has daily backup                                                                                                                                                                                                                                                                                         |
+----------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `<b>`{=html}Scratch space`</b>`{=html} | - For active or temporary (scratch) storage.                                                                                                                                                                                                                                                               |
|                                        | - Not allocated.                                                                                                                                                                                                                                                                                           |
|                                        | - Large fixed [quota](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "quota"){.wikilink} per user.                                                                                                                                                                                             |
|                                        | - Inactive data will be [purged](https://docs.alliancecan.ca/Scratch_purging_policy "purged"){.wikilink}.                                                                                                                                                                                                                              |
+----------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `<b>`{=html}Project space`</b>`{=html} | \|                                                                                                                                                                                                                                                                                                         |
|                                        |                                                                                                                                                                                                                                                                                                            |
|                                        | - Large adjustable [quota](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "quota"){.wikilink} per project.                                                                                                                                                                                     |
|                                        | - Has daily backup.                                                                                                                                                                                                                                                                                        |
+----------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

## Network interconnects {#network_interconnects}

Nodes are interconnected with 100Gbps Ethernet with RoCE (RDMA over Converged Ethernet) enabled.

## Scheduling

The `<b>`{=html}Vulcan`</b>`{=html} cluster uses the Slurm scheduler to run user workloads. The basic scheduling commands are similar to the other national systems.

## Software

- Module-based software stack.
- Both the standard Alliance software stack as well as cluster-specific software.
