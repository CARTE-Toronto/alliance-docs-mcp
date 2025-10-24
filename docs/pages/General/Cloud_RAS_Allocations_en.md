---
title: "Cloud RAS Allocations/en"
url: "https://docs.alliancecan.ca/wiki/Cloud_RAS_Allocations/en"
category: "General"
last_modified: "2025-09-04T19:05:18Z"
page_id: 9229
display_title: "Cloud RAS Allocations"
---

`<languages />`{=html}

*Parent page: [Cloud](https://docs.alliancecan.ca/Cloud "Cloud"){.wikilink}*

Any Digital Research Alliance of Canada user can access modest quantities of resources as soon as they have an Alliance account. The Rapid Access Service (`<b>`{=html}RAS`</b>`{=html}) allows users to experiment and to start working right away. Many research groups can meet their needs with the Rapid Access Service only. Users requiring larger resource quantities can apply to our annual [Resource Allocation Competition](https://docs.alliancecan.ca/RAC_application_guide "Resource Allocation Competition"){.wikilink} (`<b>`{=html}RAC`</b>`{=html}). Primary Investigators (PIs) with a current RAC allocation are also able to request resources via RAS.

Using cloud resources, researchers can create `<b>`{=html}`<i>`{=html}cloud instances`</b>`{=html}`</i>`{=html} (also known as `<i>`{=html}virtual machines`</i>`{=html} or `<i>`{=html}VMs`</i>`{=html}). There are two options available for cloud resources:

- `<b>`{=html}Compute instances`</b>`{=html}: These are instances that have a `<b>`{=html}limited life-time`</b>`{=html} (wall-time) and typically have `<b>`{=html}constant high CPU`</b>`{=html} requirements. They are sometimes referred to as `<i>`{=html}batch`</i>`{=html} instances. Users may need a large number of compute instances for production activities. Maximum wall-time for compute instances is `<b>`{=html}one month`</b>`{=html}. Upon reaching their life-time limit these instances will be scheduled for deactivation and their owners will be notified in order to ensure they clean up their instances and download any required data. Any grace period is subject to resources availability at that time.
- `<b>`{=html}Persistent instances`</b>`{=html}: These are instances that are meant to run `<b>`{=html}indefinitely`</b>`{=html} and would include `<b>`{=html}Web servers`</b>`{=html}, `<b>`{=html}database servers`</b>`{=html}, etc. In general, these instances provide a persistent service and use `<b>`{=html}less CPU`</b>`{=html} power than compute instances.
- `<b>`{=html}vGPU`</b>`{=html}: Arbutus currently offers H100 GPUs in a single flavor for RAS use (**g1-12vgb-c3-35gb-125**). This flavor has 12GB GPU memory, 3 vCPUs, 35GB of memory, and 125GB of ephemeral storage. Alternative GPU flavors are available for RAC recipients; researcher feedback on useful resource combinations for those new flavors is welcomed. For more information on setting up your VM to use vGPUs, see [Using cloud vGPUs](https://docs.alliancecan.ca/Using_cloud_vGPUs "Using cloud vGPUs"){.wikilink}.

## Cloud RAS resources limits {#cloud_ras_resources_limits}

+-----------------------------------------------------------------------------+------------------------------------+--------------------------+
| Attributes                                                                  | Compute instances[^1]              | Persistent instances[^2] |
+=============================================================================+====================================+==========================+
| May be requested by                                                         | PIs only                           | PIs only                 |
+-----------------------------------------------------------------------------+------------------------------------+--------------------------+
| vCPUs (see [VM flavours](https://docs.alliancecan.ca/Virtual_machine_flavors "VM flavours"){.wikilink}) | 80                                 | 25                       |
+-----------------------------------------------------------------------------+------------------------------------+--------------------------+
| vGPUs[^3]                                                                   | 1                                                             |
+-----------------------------------------------------------------------------+------------------------------------+--------------------------+
| Instances[^4]                                                               | 20                                 | 10                       |
+-----------------------------------------------------------------------------+------------------------------------+--------------------------+
| Volumes[^5]                                                                 | 2                                  | 10                       |
+-----------------------------------------------------------------------------+------------------------------------+--------------------------+
| Volume snapshots[^6]                                                        | 2                                  | 10                       |
+-----------------------------------------------------------------------------+------------------------------------+--------------------------+
| RAM (GB)                                                                    | 300                                | 50                       |
+-----------------------------------------------------------------------------+------------------------------------+--------------------------+
| Floating IP                                                                 | 2                                  | 2                        |
+-----------------------------------------------------------------------------+------------------------------------+--------------------------+
| Persistent storage (TB)                                                     | 10                                                            |
+-----------------------------------------------------------------------------+---------------------------------------------------------------+
| Object storage (TB)[^7]                                                     | 10                                                            |
+-----------------------------------------------------------------------------+---------------------------------------------------------------+
| Shared filesystem storage (TB)[^8]                                          | 10                                                            |
+-----------------------------------------------------------------------------+------------------------------------+--------------------------+
| Default duration                                                            | 1 year[^9], with 1 month wall-time | 1 year (renewable)[^10]  |
+-----------------------------------------------------------------------------+------------------------------------+--------------------------+
| Default renewal                                                             | April[^11]                         | April[^12]               |
+-----------------------------------------------------------------------------+------------------------------------+--------------------------+

## Requesting RAS {#requesting_ras}

To request RAS, please [fill out this form](https://docs.google.com/forms/d/e/1FAIpQLSeU_BoRk5cEz3AvVLf3e9yZJq-OvcFCQ-mg7p4AWXmUkd5rTw/viewform).

`<small>`{=html}

## Notes

<references/>

`</small>`{=html}

[^1]: Users may request both a compute and persistent allocation to share a single project. Storage is shared between the two allocations and is limited to 10TB/PI per storage type. PIs may request a 1-year renewal of their cloud RAS allocations an unlimited number of times; however, allocations will be given based on available resources and are not guaranteed. Requests made after January 1 will expire March of the following year and therefore may be longer than 1 year. Allocation requests made between May-December will be less than 1 year. Renewals will take effect in April.

[^2]:

[^3]:

[^4]: This is a metadata quota and not a hard limit, users can request an increase beyond these values without a RAC request.

[^5]:

[^6]:

[^7]: Currently only available at Arbutus.

[^8]:

[^9]: This is to align with the RAC allocation period of April-March.

[^10]:

[^11]:

[^12]:
