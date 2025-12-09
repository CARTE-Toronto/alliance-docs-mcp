---
title: "Béluga/en"
url: "https://docs.alliancecan.ca/wiki/B%C3%A9luga/en"
category: "General"
last_modified: "2025-10-20T17:59:56Z"
page_id: 10987
display_title: "Béluga"
---

`<languages />`{=html}

  ---------------------------------------------------------------------------------------------------------------------------------------
  Availability: March, 2019
  Login node: **beluga.alliancecan.ca**
  Globus collection: **[computecanada#beluga-dtn](https://app.globus.org/file-manager?origin_id=278b9bfe-24da-11e9-9fa2-0a06afd4a22e)**
  Data Transfer Node (rsync, scp, sftp,\...): **beluga.alliancecan.ca**
  Portal: <https://portail.beluga.calculquebec.ca/>
  ---------------------------------------------------------------------------------------------------------------------------------------

Béluga is a general purpose cluster designed for a variety of workloads and situated at the [École de technologie supérieure](http://www.etsmtl.ca/) in Montreal. The cluster is named in honour of the St. Lawrence River\'s [Beluga whale](https://en.wikipedia.org/wiki/Beluga_whale) population.

## Site-specific policies {#site_specific_policies}

By policy, Béluga\'s compute nodes cannot access the internet. If you need an exception to this rule, contact [technical support](https://docs.alliancecan.ca/Technical_support "technical support"){.wikilink} explaining what you need and why.

Crontab is not offered on Béluga.

Each job on Béluga should have a duration of at least one hour (five minutes for test jobs) and a user cannot have more than 1000 jobs, running and queued, at any given moment. The maximum duration for a job on Béluga is 7 days (168 hours).

`<span id="Stockage">`{=html}`</span>`{=html}

## Storage

+------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| HOME\                              | - Location of home directories, each of which has a small fixed quota.                                                 |
| Lustre filesystem, 105 TB of space | - You should use the `project` space for larger storage needs.                                                         |
|                                    |                                                                                                                        |
|                                    | <!-- -->                                                                                                               |
|                                    |                                                                                                                        |
|                                    | - Small fixed [quota](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "quota"){.wikilink} per user.         |
|                                    |                                                                                                                        |
|                                    | <!-- -->                                                                                                               |
|                                    |                                                                                                                        |
|                                    | - There is a daily backup of the home directories.                                                                     |
+------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| SCRATCH\                           | - Large space for storing temporary files during computations.                                                         |
| Lustre filesystem, 2.6 PB of space | - No backup system in place.                                                                                           |
|                                    |                                                                                                                        |
|                                    | <!-- -->                                                                                                               |
|                                    |                                                                                                                        |
|                                    | - Large fixed [quota](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "quota"){.wikilink} per user.         |
|                                    |                                                                                                                        |
|                                    | <!-- -->                                                                                                               |
|                                    |                                                                                                                        |
|                                    | - There is an [ automated purge](https://docs.alliancecan.ca/Scratch_purging_policy " automated purge"){.wikilink} of older files in this space.   |
+------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| PROJECT\                           | - This space is designed for sharing data among the members of a research group and for storing large amounts of data. |
| Lustre filesystem, 25 PB of space  |                                                                                                                        |
|                                    | <!-- -->                                                                                                               |
|                                    |                                                                                                                        |
|                                    | - Large adjustable [quota](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "quota"){.wikilink} per group.   |
|                                    |                                                                                                                        |
|                                    | <!-- -->                                                                                                               |
|                                    |                                                                                                                        |
|                                    | - There is a daily backup of the project space.                                                                        |
+------------------------------------+------------------------------------------------------------------------------------------------------------------------+

For transferring data via Globus, you should use the endpoint `computecanada#beluga-dtn`, while for tools like rsync and scp you can use a login node.

`<span id="Réseautique_haute_performance">`{=html}`</span>`{=html}

## High-performance interconnect {#high_performance_interconnect}

A Mellanox Infiniband EDR (100 Gb/s) network connects together all the nodes of the cluster. A central switch of 324 ports links the cluster\'s island topology with a maximum blocking factor of 5:1. The storage servers are networked with a non-blocking connection. The architecture permits multiple parallel jobs with up to 640 cores (or more) thanks to a non-blocking network. For jobs requiring greater parallelism, the blocking factor is 5:1 but even for jobs executed across several islands, the interconnection is high-performance.

## Node characteristics {#node_characteristics}

`<b>`{=html}Note: As of July 31 2025, all the nodes listed here are closed.\
`</b>`{=html} Turbo mode is activated on all compute nodes of Béluga.

+-------+-------+------------------+---------------------------------------+-------------------+--------------------------------------------------------+
| nodes | cores | available memory | CPU                                   | storage           | GPU                                                    |
+=======+=======+==================+=======================================+===================+========================================================+
| 160   | 40    | 92G or 95000M    | 2 x Intel Gold 6148 Skylake @ 2.4 GHz | 1 x SSD 480G      | \-                                                     |
+-------+-------+------------------+---------------------------------------+-------------------+--------------------------------------------------------+
| 579   | 40    | 186G or 191000M  | 2 x Intel Gold 6148 Skylake @ 2.4 GHz | 1 x SSD 480G      | \-                                                     |
+-------+       |                  |                                       +-------------------+                                                        |
| 10    |       |                  |                                       | 6 x SSD 480G      |                                                        |
+-------+-------+------------------+---------------------------------------+-------------------+--------------------------------------------------------+
| 51    | 40    | 752G or 771000M  | 2 x Intel Gold 6148 Skylake @ 2.4 GHz | 1 x SSD 480G      | \-                                                     |
+-------+       |                  |                                       +-------------------+                                                        |
| 2     |       |                  |                                       | 6 x SSD 480G      |                                                        |
+-------+-------+------------------+---------------------------------------+-------------------+--------------------------------------------------------+
| 172   | 40    | 186G or 191000M  | 2 x Intel Gold 6148 Skylake @ 2.4 GHz | 1 x NVMe SSD 1.6T | 4 x NVidia V100SXM2 (16G memory), connected via NVLink |
+-------+-------+------------------+---------------------------------------+-------------------+--------------------------------------------------------+

- To get a larger `$SLURM_TMPDIR` space, a job can be submitted with `--tmp=xG`, where `x` is a value between 350 and 2490.

## Monitoring jobs {#monitoring_jobs}

To maximize the use of resources and reduce your waiting times in the queue, you can monitor your CPU and GPU past or current compute tasks in real time in the [portal](https://portail.beluga.calculquebec.ca/).

For each job you can monitor

- the use of compute cores,
- the use of memory,
- the use of GPUs.

When compute resources are little or not used at all, it is important to use the allocated resources and to adjust your requests. For example, if you request four cores (CPUs) but only use one, you must adjust your submission file accordingly.
