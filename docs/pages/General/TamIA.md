---
title: "TamIA/en"
url: "https://docs.alliancecan.ca/wiki/TamIA/en"
category: "General"
last_modified: "2025-12-03T21:24:01Z"
page_id: 28130
display_title: "TamIA"
---

`<languages />`{=html}

  -------------------------------------------------------------------------------------------------------------------------------------
  Availability : **March 31, 2025**
  Login node : **tamia.alliancecan.ca**
  Globus collection : [TamIA\'s Globus v5 Server](https://app.globus.org/file-manager?origin_id=72c3bca0-9281-4742-b066-333ba0fdef72)
  Data transfer node (rsync, scp, sftp,\...) : **tamia.alliancecan.ca**
  Portal : <https://portail.tamia.ecpia.ca/>
  -------------------------------------------------------------------------------------------------------------------------------------

tamIA is a cluster dedicated to artificial intelligence for the Canadian scientific community. Located at [Université Laval](http://www.ulaval.ca/), tamIA is co-managed with [Mila](https://mila.quebec/) and [Calcul Québec](https://calculquebec.ca/). The cluster is named for the [eastern chipmunk](https://en.wikipedia.org/wiki/Tamias), a common species found in eastern North America.

tamIA is part of [PAICE, the Pan-Canadian AI Compute Environment](https://alliancecan.ca/en/services/advanced-research-computing/pan-canadian-ai-compute-environment-paice).

## Site-specific policies {#site_specific_policies}

-   By policy, tamIA\'s compute nodes cannot access the internet. If you need an exception to this rule, contact [technical support](https://docs.alliancecan.ca/Technical_support "wikilink") explaining what you need and why.
-   `crontab` is not offered on tamIA.
-   Please note that the **[VSCode IDE](https://code.visualstudio.com/)** is `<b>`{=html}forbidden`</b>`{=html} on the `<b>`{=html}login nodes`</b>`{=html} due to its heavy footprint. It is still authorized on the compute nodes.
-   Each job should be at least one hour long (at least five minutes for test jobs) and you can\'t have more than 1000 jobs (running and pending) at the same time.
-   The maximum duration of a job is one day (24 hours).
-   Each job must use all 4 GPUs on all nodes it is allocated. Jobs are allocated whole nodes.

```{=html}
<div class="mw-translate-fuzzy">
```
## Access

To access the cluster, each researcher must complete [an access request in the CCDB](https://ccdb.alliancecan.ca/me/access_services). Access to the cluster may take up to one hour after completing the access request is sent.

```{=html}
</div>
```
Eligible principal investigators are members of an AIP-type RAP (prefix `aip-`).

The procedure for sponsoring other researchers is as follows:

-   On the **[CCDB home page](https://ccdb.alliancecan.ca/)**, go to the *Resource Allocation Projects* table
-   Look for the RAPI of the `aip-` project and click on it to be redirected to the RAP management page
-   At the bottom of the RAP management page, click on **Manage RAP memberships**
-   To add a new member, go to *Add Members* and enter the CCRI of the user you want to add.

The cluster can only be reached from Canada.

## Storage

+--------------------+--------------------------------------------------------------------------------------------------------------------------+
| HOME\              | -   Location of home directories, each of which has a small fixed quota.                                                 |
| Lustre file system | -   You should use the `project` space for larger storage needs.                                                         |
|                    | -   Small per user [quota](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "wikilink").                       |
|                    | -   There is currently no backup of the home directories. (ETA Summer 2025)                                              |
+--------------------+--------------------------------------------------------------------------------------------------------------------------+
| SCRATCH\           | -   Large space for storing temporary files during computations.                                                         |
| Lustre file system | -   No backup system in place.                                                                                           |
|                    | -   Large [quota](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "wikilink") per user.                       |
|                    | -   There is an [automated purge](https://docs.alliancecan.ca/Scratch_purging_policy "wikilink") of older files in this space.                       |
+--------------------+--------------------------------------------------------------------------------------------------------------------------+
| PROJECT\           | -   This space is designed for sharing data among the members of a research group and for storing large amounts of data. |
| Lustre file system | -   Large and adjustable per group [quota](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "wikilink").       |
|                    | -   There is currently no backup of the home directories. (ETA Summer 2025)                                              |
+--------------------+--------------------------------------------------------------------------------------------------------------------------+

For transferring data via [Globus](https://docs.alliancecan.ca/Globus "wikilink"), you should use the endpoint specified at the top of this page, while for tools like [rsync](https://docs.alliancecan.ca/Transferring_data#Rsync "wikilink") and [scp](https://docs.alliancecan.ca/Transferring_data#SCP "wikilink") you can use a login node.

## High-performance interconnect {#high_performance_interconnect}

The [InfiniBand](https://fr.wikipedia.org/wiki/Bus_InfiniBand) [NVIDIA NDR](https://www.nvidia.com/en-us/networking/quantum2/) network links together all of the nodes of the cluster. Each H100 GPU is connected to a single NDR200 port through an NVIDIA ConnectX-7 HCA. Eeach GPU server has 4 NDR200 ports connected to the InfiniBand fabric.

The InfiniBand network is non-blocking for compute servers and is composed of two levels of switches in a fat-tree topology. Storage and management nodes are connected via four 400Gb/s connections to the network core.

## Node characteristics {#node_characteristics}

  nodes   cores   available memory   CPU                                                                                                                                                                             storage             GPU
  ------- ------- ------------------ ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------- --------------------------------------------------------------
  42      48      512GB              2 x [Intel Xeon Gold 6442Y 2,6 GHz, 24C](https://www.intel.com/content/www/us/en/products/sku/232380/intel-xeon-gold-6442y-processor-60m-cache-2-60-ghz/specifications.html)    1 x SSD de 7.68TB   4 x NVIDIA HGX H100 SXM 80GB HBM3 700W, connected via NVLink
  4       64      512GB              2 x [Intel Xeon Gold 6438M 2.2G, 32C/64T](https://www.intel.com/content/www/us/en/products/sku/232398/intel-xeon-gold-6438m-processor-60m-cache-2-20-ghz/specifications.html)   1 x SSD de 7.68TB   none

### Software environments {#software_environments}

[`StdEnv/2023`](https://docs.alliancecan.ca/Standard_software_environments/fr "wikilink") is the standard environment on tamIA.

`<span id="Suivi_de_vos_tâches">`{=html}`</span>`{=html}

## Monitoring jobs {#monitoring_jobs}

From the tamIA [portal](https://portail.tamia.ecpia.ca/), you can monitor your jobs using CPUs and GPUs `<b>`{=html}in real time`</b>`{=html} or examine jobs that have run in the past. This can help you to optimize resource usage and shorten wait time in the queue.

You can monitor your usage of

-   compute nodes,
-   memory,
-   GPU.

It is important that you use the allocated resources and to correct your requests when compute resources are less used or not used at all. For example, if you request 4 cores (CPUs) but use only one, you should adjust the script file accordingly.
