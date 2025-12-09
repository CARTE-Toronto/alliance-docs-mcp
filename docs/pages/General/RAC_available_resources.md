---
title: "RAC available resources/en"
url: "https://docs.alliancecan.ca/wiki/RAC_available_resources/en"
category: "General"
last_modified: "2025-09-11T18:45:39Z"
page_id: 29764
display_title: "RAC available resources"
---

`<languages />`{=html} Below is the list of resources available for the **Resource Allocation Competition** **2026**.

Resources in the clusters and in the cloud are organized by subsystems*.* Each subsystem will only show the resources that it has available. For example, the *trillium-storage* subsystem will only show /project storage as /nearline storage is available in the *hpss-storage* subsystem; *trillium-compute* will only show CPU (and memory), etc.

***Important: There should be no discrepancies between the resources requested in the document attached to your application and the online form. In case of discrepancy, what is requested in the online form will prevail.***

+-------------------------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
| **System**                                            | **Sub-system (as shown on CCDB)** | **Available resources for each sub-system**                                                                                                       | **Backup storage?** |
+-------------------------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
| [Arbutus](https://docs.alliancecan.ca/Cloud "Arbutus"){.wikilink} cloud           | arbutus-compute-cloud             | VCPU, VGPU, RAM, local ephemeral disk, volumes, snapshots, floating IPs, volume and snapshot storage, shared filesystem storage\*, object storage | No                  |
|                                                       +-----------------------------------+                                                                                                                                                   |                     |
|                                                       | arbutus-persistent-cloud          |                                                                                                                                                   |                     |
|                                                       +-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
|                                                       | arbutus-dcache                    | dCache storage                                                                                                                                    | No                  |
+-------------------------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
| [Rorqual](https://docs.alliancecan.ca/Rorqual/en "Rorqual"){.wikilink} cluster    | rorqual-compute                   | CPU                                                                                                                                               | No                  |
|                                                       +-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
|                                                       | rorqual-gpu                       | GPU                                                                                                                                               | No                  |
|                                                       +-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
|                                                       | rorqual-storage                   | Project storage, Nearline storage                                                                                                                 | Yes                 |
+-------------------------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
| [Béluga](https://docs.alliancecan.ca/Cloud/en "Béluga"){.wikilink} cloud          | beluga-compute-cloud              | VCPU, RAM, local ephemeral disk, volumes, snapshots, floating IPs, volume and snapshot storage                                                    | No                  |
|                                                       +-----------------------------------+                                                                                                                                                   |                     |
|                                                       | beluga-persistent-cloud           |                                                                                                                                                   |                     |
+-------------------------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
| [Fir](https://docs.alliancecan.ca/Fir/en "Fir"){.wikilink} cluster                | fir-compute                       | CPU                                                                                                                                               | No                  |
|                                                       +-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
|                                                       | fir-gpu                           | GPU                                                                                                                                               | No                  |
|                                                       +-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
|                                                       | fir-storage                       | Project storage, Nearline storage, dCache storage                                                                                                 | Yes                 |
+-------------------------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
| Fir cloud                                             | fir-persistent-cloud              | VCPU, RAM, local ephemeral disk, volumes, snapshots, floating IPs, volume and snapshot storage, object storage                                    | No                  |
|                                                       +-----------------------------------+                                                                                                                                                   |                     |
|                                                       | fir-compute-cloud                 |                                                                                                                                                   |                     |
+-------------------------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
| [Nibi](https://docs.alliancecan.ca/Nibi "Nibi"){.wikilink} cluster                | nibi-compute                      | CPU                                                                                                                                               | No                  |
|                                                       +-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
|                                                       | nibi-gpu                          | GPU                                                                                                                                               | No                  |
|                                                       +-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
|                                                       | nibi-storage                      | Project storage, Nearline storage, dCache storage                                                                                                 | Yes                 |
+-------------------------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
| [Nibi](https://docs.alliancecan.ca/Cloud "Nibi"){.wikilink} cloud                 | nibi-persistent-cloud             | VCPU, RAM, local ephemeral disk, volumes, snapshots, floating IPs, volume and snapshot storage                                                    | No                  |
+-------------------------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
| [Narval](https://docs.alliancecan.ca/Narval/en "Narval"){.wikilink} cluster       | narval-compute                    | CPU                                                                                                                                               | No                  |
|                                                       +-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
|                                                       | narval-gpu                        | GPU                                                                                                                                               | No                  |
|                                                       +-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
|                                                       | narval-storage                    | Project storage                                                                                                                                   | Yes                 |
+-------------------------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
| [Trillium](https://docs.alliancecan.ca/Trillium/fr "Trillium"){.wikilink} cluster | trillium-compute                  | CPU                                                                                                                                               | No                  |
|                                                       +-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
|                                                       | trillium-gpu                      | GPU                                                                                                                                               | No                  |
|                                                       +-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
|                                                       | trillium-storage                  | Project storage                                                                                                                                   | Yes                 |
+-------------------------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
| HPSS                                                  | hpss-storage                      | Nearline storage                                                                                                                                  | No                  |
+-------------------------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+

*\* The shared filesystem storage is backed up.*

## **How to request resources in the online RAC application form** {#how_to_request_resources_in_the_online_rac_application_form}

On the CCDB portal, go to the *Resource Request* section of the corresponding Resources for Research Groups (RRG) or Research Platforms and Portals (RPP) online application form.

The dropdown menu under *New resource request* will show the list of all systems and subsystems with the corresponding resources available for allocation. To indicate that your request can be allocated in any resource and that you do not have a preference for a particular one, please check the option \"I have to select a system but I don't mind receiving an allocation on any other suitable one\" in the *Reason for selecting system* section.

**1. Requesting HPC resources:** compute and storage resources are shown as different subsystems on CCDB following this convention:

- system-compute (e.g. *rorqual-compute*): select these to request CPU resources. Please pay special attention to your memory requirements as this will be taken into account for final allocations.
- system-gpu (e.g. *rorqual-gpu*): select these to request GPU resources.
- system-storage (e.g. *rorqual-storage*): select these to request storage resources on a particular system. Each subsystem will only show the storage resources (/project, /nearline, etc.) available for that particular subsystem.

For example, if you want to request both CPU resources and /project storage resources on the Rorqual cluster, you will have to fill in two separate requests: one with the rorqual-compute subsystem for core years and memory, and another one with the rorqual-storage subsystem for /project storage (in TB).

**2. Requesting Cloud resources:** If you need to request cloud resources in more than one location, then you will have to make one request for each location. On CCDB, cloud resources follow this convention:

- system-compute-cloud or system-persistent-cloud: you can select compute or persistent VMs on the Arbutus, Fir and Béluga clouds, but only persistent VMs on the NIbi cloud. Each cloud resource will **only** show the specific \"flavours\" of VMs available for that particular site.

Please email allocations@tech.alliancecan.ca with any questions on how to request resources.
