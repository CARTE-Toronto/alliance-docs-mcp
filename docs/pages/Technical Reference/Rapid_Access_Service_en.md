---
title: "Rapid Access Service/en"
url: "https://docs.alliancecan.ca/wiki/Rapid_Access_Service/en"
category: "Technical Reference"
last_modified: "2025-08-13T21:27:36Z"
page_id: 29782
display_title: "Rapid Access Service"
---

`<languages />`{=html}

## HPC resources {#hpc_resources}

### Storage

Some [storage resources](https://docs.alliancecan.ca/Storage_and_file_management "storage resources"){.wikilink} are made available in the Default RAP to PIs and their sponsored users immediately after creating an Alliance account on CCDB and they will be ready for utilization as soon as access to the corresponding system is requested here: <https://ccdb.alliancecan.ca/me/access_systems>. Visit this page for more details about the storage resources available by default.

When the resources mentioned above are not sufficient, PIs can request additional storage resources in any General Purpose cluster without submitting a RAC application, up to a maximum of 40 TB of project storage and 100 TB of nearline storage. These resources can be requested all in one cluster or split across multiple ones, *but the total amount across all clusters must not exceed 40 TB of project storage or 100 TB of nearline storage*. Resources allocated via RAS will be available in the Default [Resource Allocation Project](https://docs.alliancecan.ca/Frequently_Asked_Questions_about_the_CCDB#Resource_Allocation_Projects_(RAP) "Resource Allocation Project"){.wikilink} (RAP).

+----------------------------+---------------------------+---------------------------+
| **Cluster**                | **Project storage**       | **Nearline storage**      |
+----------------------------+---------------------------+---------------------------+
| Fir, Nibi, Rorqual, Narval | Max 40 TB                 | Max 100 TB                |
|                            |                           |                           |
|                            | *across all clusters*     | *across all clusters*     |
+----------------------------+---------------------------+---------------------------+
| Trillium, HPSS             | RAS storage not available | RAS storage not available |
+----------------------------+---------------------------+---------------------------+

To request storage resources via RAS, PIs (not sponsored users) must send an email to support@tech.alliancecan.ca with details of the storage resources needed.

### CPU

**CPU resources are available for *opportunistic use*** to all research groups with an active Alliance account. The great majority of jobs submitted this way get executed, albeit with sometimes a lower priority than jobs submitted with a RAC allocation.

Research groups with an Alliance account needing CPU resources **may** be able to use, **on average**, up to 200 core years on each cluster with their Default RAP. Note that 200 core years is a variable target: it is not a reservation nor a cap, which means that research groups could utilize more or less than that amount depending on the shape and size of the jobs and on the overall utilization of the cluster.

Most research groups are able to meet their need for CPU resources by submitting jobs opportunistically with their Default RAP and without having to apply for the RAC. Jobs with large memory requirements may take longer to run: in those cases, applying for RAC may be the best option if the amount of CPU resources needed exceeds the minimum required to apply.

Please read about [Allocations and compute scheduling](https://docs.alliancecan.ca/Allocations_and_compute_scheduling "Allocations and compute scheduling"){.wikilink}. This information can help you better understand how jobs are scheduled in our clusters and how CPU usage is charged.

### GPU

GPU resources are available for opportunistic use to all research groups with an active Alliance account.

The demand for GPUs is increasing quickly with the advances in Artificial Intelligence. The availability of GPU resources varies greatly over the year, becoming more limited in periods preceding major conferences.

We cannot therefore guarantee any amount of resources available to each group for opportunistic use, especially during times of high demand. Users with a RAC award should be able to consistently use their allocated amount.

## Cloud resources {#cloud_resources}

Access to modest amounts of cloud resources, within the limits detailed in the [cloud RAS documentation](https://docs.alliancecan.ca/Cloud_RAS_Allocations "cloud RAS documentation"){.wikilink}, can be requested at any time. PIs must complete this [form](https://docs.google.com/forms/d/e/1FAIpQLSeU_BoRk5cEz3AvVLf3e9yZJq-OvcFCQ-mg7p4AWXmUkd5rTw/viewform) with details about cloud resource needed.

Other users with an Alliance account may also be granted access these resources.

If you have questions about accessing RAS cloud resources or need help, please contact cloud@tech.alliancecan.ca.
