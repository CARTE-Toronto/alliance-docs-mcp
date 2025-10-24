---
title: "Killarney"
url: "https://docs.alliancecan.ca/wiki/Killarney"
category: "General"
last_modified: "2025-10-23T18:11:28Z"
page_id: 27762
display_title: "Killarney"
---

`<languages />`{=html}

`<translate>`{=html}

  ----------------------------------------------------------------------
  Availability: June 9, 2025
  Login node: `<b>`{=html}killarney.alliancecan.ca`</b>`{=html}
  Globus collection: TBA
  System Status Page: <https://status.alliancecan.ca/system/Killarney>
  ----------------------------------------------------------------------

`<b>`{=html}Killarney`</b>`{=html} is a cluster dedicated to the needs of the Canadian scientific Artificial Intelligence community. `<b>`{=html}Killarney`</b>`{=html} is located at the [University of Toronto](https://www.utoronto.ca/) and is managed by the [Vector Institute](https://vectorinstitute.ai/) and [SciNet](https://www.scinethpc.ca/). It is named after the [Killarney Ontario Provincial Park](https://www.ontarioparks.ca/park/killarney), located near Georgian Bay.

This cluster is part of the Pan-Canadian AI Compute Environment (PAICE).

## Site-specific policies {#site_specific_policies}

Killarney is currently open to Vector affiliated PIs with CCAI Chairs. Further access will be announced at a later date.

## Access

To access Killarney, each researcher must [request access in the CCDB](https://ccdb.alliancecan.ca/me/access_services).

Principal Investigators must be granted an AIP-type RAP (prefix `aip-` ) by their AI Institution. For the PI to sponsor researchers in their AIP RAP, the PI must:

-   Go to the \"Resource Allocation Projects\" table on the CCDB Home page.
-   Locate the RAPI of your AIP project (with the aip- prefix) and click on it to reach the RAP management page.
-   At the bottom of the RAP management page, click on \"Manage RAP memberships.\"
-   Enter the CCRI of the user you want to add in the \"Add Members\" section.

To ensure the integrity and security of this resource, Vector enforces geo-blocking on Killarney as one of its cyber-security controls. Vector restricts access to and from countries identified in the [Government of Canada\'s Cyber Threat Assessment](https://www.cyber.gc.ca/en/guidance/national-cyber-threat-assessment-2025-2026).

## Killarney hardware specifications {#killarney_hardware_specifications}

  Performance Tier      Nodes   Model         CPU                         Cores   System Memory   GPUs per node              Total GPUs
  --------------------- ------- ------------- --------------------------- ------- --------------- -------------------------- ------------
  Standard Compute      168     Dell 750xa    2 x Intel Xeon Gold 6338    64      512 GB          4 x NVIDIA L40S 48GB       672
  Performance Compute   10      Dell XE9680   2 x Intel Xeon Gold 6442Y   48      2048 GB         8 x NVIDIA H100 SXM 80GB   80

## Storage system {#storage_system}

`<b>`{=html}Killarney`</b>`{=html}\'s storage system is an all-NVME VastData platform with a total usable capacity of 1.7PB.

+----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| `<b>`{=html}Home space`</b>`{=html}    | -   Location of /home directories.                                                                                         |
|                                        | -   Each /home directory has a small fixed [quota](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "wikilink"). |
|                                        | -   Larger requests go to the /project space.                                                                              |
|                                        | -   Has daily backup                                                                                                       |
+----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| `<b>`{=html}Scratch space`</b>`{=html} | -   For active or temporary (scratch) storage.                                                                             |
|                                        | -   Large fixed [quota](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "wikilink") per user.                   |
|                                        | -   Inactive data will be [purged](https://docs.alliancecan.ca/Scratch_purging_policy "wikilink").                                                     |
+----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| `<b>`{=html}Project space`</b>`{=html} | -   Large adjustable [quota](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "wikilink") per project.           |
|                                        | -   Has daily backup.                                                                                                      |
+----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+

## Network interconnects {#network_interconnects}

Standard Compute nodes are interconnected with Infiniband HDR100 for 100Gbps throughput, while Performance Compute nodes are connected with 2 x HDR 200 for 400Gbps aggregate throughput.

## Scheduling

The `<b>`{=html}Killarney`</b>`{=html} cluster uses the Slurm scheduler to run user workloads. The basic scheduling commands are similar to the other national systems.

## Software

-   Module-based software stack.
-   Both the standard Alliance software stack as well as cluster-specific software.

`</translate>`{=html}
